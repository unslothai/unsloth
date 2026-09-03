// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #8867: the bar rendered only once a token count existed, so a local model's context window was
// invisible until a reply came back. An earlier version of this file pinned source text only, and
// a build that fell back to 0 instead of null still passed every assertion.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { deriveContextUsageBar } from "../src/features/chat/lib/context-usage-bar-state.ts";
import { hasKnownContextWindow } from "../src/features/chat/lib/context-window-known.ts";

const RESIDENT = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF";

const base = {
  loadedContextLength: 32768,
  modelLoading: false,
  isExternalModel: false,
  residentCheckpoint: RESIDENT as string | null | undefined,
};

test("a resident GGUF's window is known before the first turn", () => {
  assert.equal(hasKnownContextWindow(base), true);
});

// the window belongs to the outgoing model until the load response lands
test("a load in flight has no window to name", () => {
  assert.equal(hasKnownContextWindow({ ...base, modelLoading: true }), false);
});

// selecting an API model nulls loadedContextLength, so a stale length must be refused on its own
test("an API model shows no window even with a stale length in the store", () => {
  assert.equal(hasKnownContextWindow({ ...base, isExternalModel: true }), false);
});

test("a non-GGUF local model has no window either", () => {
  assert.equal(
    hasKnownContextWindow({ ...base, loadedContextLength: null }),
    false,
  );
});

test("a model evicted for an image load has no window", () => {
  assert.equal(
    hasKnownContextWindow({ ...base, residentCheckpoint: null }),
    false,
  );
});

// same rule as chatModelLoaded: the first /status read has not landed yet
test("residency not yet read still names the window", () => {
  assert.equal(
    hasKnownContextWindow({ ...base, residentCheckpoint: undefined }),
    true,
  );
});

// the reported case: a GGUF is resident, its window is known, nothing has been counted
test("an uncounted chat names the window and claims no usage", () => {
  const state = deriveContextUsageBar({ used: null, total: 32768 });
  assert.ok(state);
  assert.equal(state.face, "— / 32.8k");
  assert.equal(state.totalRowName, "Context window");
  assert.equal(state.totalRowValue, "32,768");
  // null, not 0: an unmeasured prompt must not read as 0% of the window
  assert.equal(state.percent, null);
  assert.match(state.label, /usage not counted yet/);
});

// the mutation the old source-regex tests could not see
test("a counted zero is not the same as an uncounted chat", () => {
  const state = deriveContextUsageBar({ used: 0, total: 32768 });
  assert.ok(state);
  assert.equal(state.face, "0 / 32.8k");
  assert.equal(state.percent, 0);
  assert.equal(state.totalRowName, "Total");
});

test("a counted chat states the ratio", () => {
  const state = deriveContextUsageBar({
    used: 4096,
    total: 32768,
    promptTokens: 4096,
    completionTokens: 0,
  });
  assert.ok(state);
  assert.equal(state.face, "4.1k / 32.8k");
  assert.equal(state.percent, 12.5);
  assert.equal(state.totalRowValue, "4,096 / 32,768");
  assert.equal(state.hasUsageDetails, true);
});

// a prompt already over the window pins at 100 rather than overflowing the fill
test("usage past the window clamps to 100 percent", () => {
  assert.equal(deriveContextUsageBar({ used: 40000, total: 32768 })?.percent, 100);
});

// llama.cpp stops at the window; MLX runs straight past it, so the advice differs, and
// which side of the limit a chat is on cannot be read from the clamped percent
test("the limit advice follows the backend and the unclamped ratio", () => {
  const at = { used: 40000, total: 32768 };
  assert.equal(deriveContextUsageBar(at)?.advice, "stops-at-limit");
  // The MLX branches are about the ratio, so they state the bound they assume: an
  // unconfirmed window advises on its own terms (see mlx-context-helpers).
  assert.equal(
    deriveContextUsageBar({ ...at, isMlx: true, contextEnforced: true })?.advice,
    "mlx-past-limit",
  );
  assert.equal(
    deriveContextUsageBar({
      used: 30000,
      total: 32768,
      isMlx: true,
      contextEnforced: true,
    })?.advice,
    "mlx-near-limit",
  );
  assert.equal(deriveContextUsageBar({ used: 4096, total: 32768 })?.advice, "none");
  // no window and no count: nothing to advise against
  assert.equal(deriveContextUsageBar({ used: 40000, total: null })?.advice, "none");
});

// external providers: usage is known, the window is not
test("an unknown window shows a bare token count and no ratio", () => {
  const state = deriveContextUsageBar({ used: 4096, total: null });
  assert.ok(state);
  assert.equal(state.face, "4.1k tokens");
  assert.equal(state.percent, null);
  assert.equal(state.totalRowName, "Total tokens");
});

test("no window and no count renders nothing", () => {
  assert.equal(deriveContextUsageBar({ used: null, total: null }), null);
  assert.equal(deriveContextUsageBar({ used: 0, total: null }), null);
});

// the divider would otherwise float above an empty region
test("an uncounted chat reports no per-turn rows", () => {
  assert.equal(
    deriveContextUsageBar({ used: null, total: 32768 })?.hasUsageDetails,
    false,
  );
});

test("the header renders the bar on the window alone, with usage optional", () => {
  const page = readFileSync(
    new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    page,
    /view\.mode === "single" && \(contextUsage \|\| contextWindowKnown\)/,
  );
  assert.match(page, /used=\{contextUsage\?\.totalTokens \?\? null\}/);
});

