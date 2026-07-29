// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/// <reference types="vite/client" />

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register(new URL("./module-alias-loader.mjs", import.meta.url));

test("execution tracking forwards its captured owner to every read", async () => {
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: { clearTimeout, setTimeout },
  });
  const { createBaseExecutionRecord } = await import(
    "../src/features/recipe-studio/executions/runtime.ts"
  );
  const { trackRecipeExecution } = await import(
    "../src/features/recipe-studio/executions/tracker.ts"
  );
  const { resetTrackerCalls, trackerCalls } = await import(
    "./tracker-runtime-fixture.ts"
  );
  resetTrackerCalls();

  const ownerKey = "subject:execution-owner";
  const currentKey = "subject:current-account";
  assert.notEqual(ownerKey, currentKey);

  const initialExecution = {
    ...createBaseExecutionRecord({
      recipeId: "recipe-A",
      kind: "full",
      rows: 10,
      currentSignature: "signature-A",
    }),
    jobId: "job-A",
  };
  const result = await trackRecipeExecution({
    label: "Full run",
    kind: "full",
    rows: 10,
    jobId: "job-A",
    expectedSubjectKey: ownerKey,
    initialExecution,
    notify: false,
    onUpsert() {
      // State rendering is outside this forwarding contract.
    },
    onSetPreviewErrors() {
      // Preview rendering is outside this forwarding contract.
    },
  });

  assert.deepEqual(result, { success: true, terminal: true });
  assert.equal(trackerCalls.stream.length, 1);
  assert.ok(trackerCalls.status.length > 0);
  assert.equal(trackerCalls.analysis.length, 1);
  assert.equal(trackerCalls.dataset.length, 1);

  for (const call of Object.values(trackerCalls).flat()) {
    assert.equal(call.jobId, "job-A");
    assert.equal(call.options.expectedSubjectKey, ownerKey);
    assert.notEqual(call.options.expectedSubjectKey, currentKey);
  }
});
