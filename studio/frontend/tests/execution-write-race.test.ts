// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/// <reference types="vite/client" />

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import type { RecipeExecutionRecord } from "../src/features/recipe-studio/execution-types.ts";

register(new URL("./module-alias-loader.mjs", import.meta.url));

test("a queued job ID survives navigation behind an in-flight base write", async () => {
  const { saveRecipeExecution } = await import(
    "../src/features/recipe-studio/data/executions-db.ts"
  );
  const {
    persistedJobId,
    releaseFirstWrite,
    resetExecutionWriteFixture,
    setAuthSubject,
    waitForFirstWrite,
    writes,
  } = await import("./execution-write-race-fixture.ts");
  resetExecutionWriteFixture();

  let current = true;
  const owner = {
    subjectKey: "subject:a",
    recipeId: "recipe-1",
    generation: 1,
    isCurrent: () => current,
  };
  const base: RecipeExecutionRecord = {
    id: "execution-1",
    recipeId: "recipe-1",
    jobId: null,
    kind: "full",
    run_name: null,
    status: "pending",
    rows: 1,
    recipeSignature: "signature-1",
    stage: null,
    current_column: null,
    completed_columns: [],
    progress: null,
    column_progress: null,
    batch: null,
    source_progress: null,
    model_usage: null,
    lastEventId: null,
    artifact_path: null,
    log_lines: [],
    dataset: [],
    datasetTotal: 0,
    datasetPage: 1,
    datasetPageSize: 20,
    analysis: null,
    processor_artifacts: null,
    error: null,
    createdAt: 1_700_000_000_000,
    finishedAt: null,
  };

  const baseSave = saveRecipeExecution(base, owner);
  await waitForFirstWrite();
  const jobSave = saveRecipeExecution(
    { ...base, jobId: "job-1", status: "active" },
    owner,
  );
  current = false;
  releaseFirstWrite();

  const results = await Promise.allSettled([baseSave, jobSave]);
  assert.deepEqual(
    results.map((result) => result.status),
    ["rejected", "fulfilled"],
  );
  assert.equal(writes.length, 3, "the durable retry must converge after 409");
  assert.equal(persistedJobId(), "job-1");

  resetExecutionWriteFixture();
  current = true;
  const otherBase = { ...base, id: "execution-2" };
  const otherBaseSave = saveRecipeExecution(otherBase, owner);
  await waitForFirstWrite();
  const otherJobSave = saveRecipeExecution(
    { ...otherBase, jobId: "job-2", status: "active" },
    owner,
  );
  setAuthSubject("subject:b");
  current = false;
  releaseFirstWrite();

  const switchedResults = await Promise.allSettled([
    otherBaseSave,
    otherJobSave,
  ]);
  assert.deepEqual(
    switchedResults.map((result) => result.status),
    ["rejected", "rejected"],
  );
  assert.equal(writes.length, 1, "an account switch must block the durable write");
  assert.equal(persistedJobId(), null);
});
