// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { deriveContextUsageBar } from "../src/features/chat/lib/context-usage-bar-state.ts";
import { hasKnownContextWindow } from "../src/features/chat/lib/context-window-known.ts";

import { readSrc } from "./helpers/kit.ts";

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

test("a load in flight has no window to name", () => {
  assert.equal(hasKnownContextWindow({ ...base, modelLoading: true }), false);
});

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

test("residency not yet read still names the window", () => {
  assert.equal(
    hasKnownContextWindow({ ...base, residentCheckpoint: undefined }),
    true,
  );
});

test("an uncounted chat names the window and claims no usage", () => {
  const state = deriveContextUsageBar({ used: null, total: 32768 });
  assert.ok(state);
  assert.equal(state.face, "— / 32.8k");
  assert.equal(state.totalRowName, "Context window");
  assert.equal(state.totalRowValue, "32,768");
  // null, not 0: an unmeasured prompt must not read as 0%.
  assert.equal(state.percent, null);
  assert.match(state.label, /usage not counted yet/);
});

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

test("usage past the window clamps to 100 percent", () => {
  assert.equal(deriveContextUsageBar({ used: 40000, total: 32768 })?.percent, 100);
});

// llama.cpp stops at the window, MLX runs past it, so advice uses the unclamped ratio.
test("the limit advice follows the backend and the unclamped ratio", () => {
  const at = { used: 40000, total: 32768 };
  assert.equal(deriveContextUsageBar(at)?.advice, "stops-at-limit");
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
  assert.equal(deriveContextUsageBar({ used: 40000, total: null })?.advice, "none");
});

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

test("an uncounted chat reports no per-turn rows", () => {
  assert.equal(
    deriveContextUsageBar({ used: null, total: 32768 })?.hasUsageDetails,
    false,
  );
});

test("the header renders the bar on the window alone, with usage optional", () => {
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(
    page,
    /showContextWindowUsage &&\s*view\.mode === "single" &&\s*\(contextUsage \|\| contextWindowKnown\)/,
  );
  assert.match(page, /used=\{contextUsage\?\.totalTokens \?\? null\}/);
});

