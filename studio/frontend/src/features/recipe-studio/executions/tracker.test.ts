// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  status: vi.fn(),
  stream: vi.fn(),
}));
vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  getRecipeJobStatus: mocks.status,
  streamRecipeJobEvents: mocks.stream,
  getRecipeJobAnalysis: vi.fn(),
  getRecipeJobDataset: vi.fn(),
}));
vi.mock("@/shared/toast", () => ({ toastError: vi.fn(), toastSuccess: vi.fn() }));

import { RecipeApiError } from "../api";
import type { RecipeExecutionRecord } from "../execution-types";
import { trackRecipeExecution } from "./tracker";

const staleExecution: RecipeExecutionRecord = {
  id: "run-1",
  recipeId: "recipe-1",
  jobId: "old-process-job",
  kind: "full",
  run_name: "old run",
  status: "running",
  rows: 1,
  recipeSignature: "signature",
  stage: "running",
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
  createdAt: 1,
  finishedAt: null,
};

describe("rehydrated execution reconciliation", () => {
  it("terminalizes a process-local job missing after backend restart", async () => {
    mocks.stream.mockRejectedValue(new RecipeApiError(404, "missing"));
    mocks.status.mockRejectedValue(new RecipeApiError(404, "missing"));
    const onUpsert = vi.fn();
    await expect(
      trackRecipeExecution({
        label: "Run",
        kind: "full",
        rows: 1,
        jobId: "old-process-job",
        initialExecution: staleExecution,
        notify: false,
        onUpsert,
        onSetPreviewErrors: vi.fn(),
      }),
    ).resolves.toEqual({ success: false, terminal: true });
    expect(onUpsert).toHaveBeenLastCalledWith(
      expect.objectContaining({
        status: "error",
        finishedAt: expect.any(Number),
        error: expect.stringContaining("previous backend session"),
      }),
    );
  });
});
