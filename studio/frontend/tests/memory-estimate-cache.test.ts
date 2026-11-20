// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import type * as HookModule from "../src/features/model-picker/hooks/use-memory-estimate.ts";
import type { MemoryEstimate } from "../src/features/model-picker/api/memory-estimate.ts";
import * as estimateContext from "../src/features/model-picker/model-config/estimate-context.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

let fetches = 0;
const hook = loadWithStubs<typeof HookModule>(
  new URL("../src/features/model-picker/hooks/use-memory-estimate.ts", import.meta.url),
  {
    react: React,
    "@/hooks/use-system": { fetchSystemInfo: async () => ({ memory_refreshed: true }) },
    "../api/memory-estimate": {
      fetchMemoryEstimate: async () => {
        fetches += 1;
        return { available: true };
      },
    },
    "../model-config/estimate-context": estimateContext,
  },
);

const estimate = { available: true, totalBytes: 7 } as MemoryEstimate;

function firstPaint(request: { modelPath: string; nCtx?: number }, refreshMemory = false) {
  let seen: ReturnType<typeof hook.useMemoryEstimate> | null = null;
  const Probe = () => {
    seen = hook.useMemoryEstimate(request, { refreshMemory });
    return null;
  };
  renderToStaticMarkup(React.createElement(Probe));
  return seen!;
}

test("a model with no answer yet paints as loading, so the row can hold its shape", () => {
  const state = firstPaint({ modelPath: "never-seen" });
  assert.equal(state.estimate, null);
  assert.equal(state.loading, true);
});

test("reopening the panel paints the last answer on the first frame, marked for re-pricing", () => {
  const request = { modelPath: "org/model-GGUF", nCtx: 0 };
  hook.rememberMemoryEstimate(hook.estimateKey(request)!, estimate);
  const state = firstPaint(request);
  assert.equal(state.estimate, estimate);
  assert.equal(state.loading, false);
  assert.equal(state.stale, true);
  // Another setting is another key.
  assert.equal(firstPaint({ ...request, nCtx: 8192 }).estimate, null);
  // The resident probe has to re-read memory, so it never starts from the cache.
  assert.equal(firstPaint(request, true).estimate, null);
  assert.equal(fetches, 0);
});

test("the cache keeps the most recent 32 answers", () => {
  for (let i = 0; i < 40; i++) hook.rememberMemoryEstimate(`k${i}`, estimate);
  assert.equal(hook.cachedMemoryEstimate("k7"), null);
  assert.equal(hook.cachedMemoryEstimate("k8"), estimate);
  hook.rememberMemoryEstimate("k8", estimate);
  hook.rememberMemoryEstimate("k40", estimate);
  assert.equal(hook.cachedMemoryEstimate("k8"), estimate, "a re-used key moves to the back");
  assert.equal(hook.cachedMemoryEstimate("k9"), null);
});
