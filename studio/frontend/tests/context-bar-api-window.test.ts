// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  contextBarWindow,
  deriveContextUsageBar,
} from "../src/features/chat/lib/context-usage-bar-state.ts";

test("a local model keeps its loaded window", () => {
  assert.equal(
    contextBarWindow({
      isExternalModel: false,
      loadedContextLength: 32768,
      reportedWindow: 260000,
      catalogWindow: 128000,
    }),
    32768,
  );
});

test("an API model never shows the resident local model's window", () => {
  const base = { isExternalModel: true, loadedContextLength: 32768 };
  assert.equal(
    contextBarWindow({ ...base, reportedWindow: 260000, catalogWindow: 128000 }),
    260000,
  );
  assert.equal(
    contextBarWindow({ ...base, reportedWindow: undefined, catalogWindow: 128000 }),
    128000,
  );
  assert.equal(
    contextBarWindow({ ...base, reportedWindow: undefined, catalogWindow: null }),
    null,
  );
});

test("an API model's window shows no Context Length advice", () => {
  const near = { used: 250000, total: 260000 };
  assert.equal(deriveContextUsageBar(near)?.advice, "stops-at-limit");
  const api = deriveContextUsageBar({ ...near, providerWindow: true });
  assert.equal(api?.advice, "none");
  assert.equal(api?.face, "250.0k / 260.0k");
});

test("an API model with no window shows its count without a total", () => {
  const state = deriveContextUsageBar({ used: 37, total: null, providerWindow: true });
  assert.equal(state?.face, "37 tokens");
  assert.equal(state?.percent, null);
});
