// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { beforeEach, describe, expect, it, vi } from "vitest";
import type { RecipeExecutionRecord } from "../execution-types";

const mocks = vi.hoisted(() => ({
  list: vi.fn(),
  upsert: vi.fn(),
  subjectKey: vi.fn(() => "subject:a"),
}));
vi.mock("@/features/auth", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/features/auth")>()),
  getAuthSubjectKey: mocks.subjectKey,
}));
vi.mock("@/features/user-assets", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/features/user-assets")>()),
  listServerRecipeExecutions: mocks.list,
  upsertServerRecipeExecution: mocks.upsert,
}));

import { UserAssetApiError } from "@/features/user-assets";
import {
  listRecipeExecutions,
  saveRecipeExecution,
  serializeExecutionMetadata,
} from "./executions-db";

function execution(
  overrides: Partial<RecipeExecutionRecord> = {},
): RecipeExecutionRecord {
  return {
    id: "run-1",
    recipeId: "recipe-1",
    jobId: "job-1",
    kind: "full",
    run_name: "run",
    status: "running",
    rows: 4,
    recipeSignature: "signature",
    stage: "generate",
    current_column: null,
    completed_columns: ["prompt"],
    progress: null,
    column_progress: null,
    batch: null,
    source_progress: null,
    model_usage: null,
    lastEventId: 9,
    datasetTotal: 4,
    analysis: null,
    error: null,
    createdAt: 10,
    finishedAt: null,
    artifact_path: "/private/artifact",
    log_lines: ["private log"],
    dataset: [{ private: "dataset row" }],
    datasetPage: 1,
    datasetPageSize: 50,
    processor_artifacts: { private: "processor path" },
    ...overrides,
  };
}

describe("execution persistence", () => {
  beforeEach(() => {
    mocks.subjectKey.mockReturnValue(`subject:${crypto.randomUUID()}`);
    mocks.list.mockReset();
    mocks.upsert.mockReset();
  });

  it("converges a terminal write after two consecutive CAS conflicts", async () => {
    const first = {
      ...serializeExecutionMetadata(execution({ lastEventId: 4 })),
      revision: 2,
      updatedAt: 20,
    };
    const second = { ...first, lastEventId: 5, revision: 3 };
    mocks.upsert
      .mockRejectedValueOnce(
        new UserAssetApiError(409, { current: first }),
      )
      .mockRejectedValueOnce(
        new UserAssetApiError(409, { current: second }),
      )
      .mockResolvedValueOnce({ ...second, status: "completed", revision: 4 });

    await saveRecipeExecution(
      execution({ id: "terminal-converges", status: "completed", lastEventId: 6 }),
    );
    expect(mocks.upsert).toHaveBeenCalledTimes(3);
    expect(mocks.upsert.mock.calls[2][0]).toMatchObject({ revision: 3 });
  });

  it("does not reuse revisions across authenticated subject transitions", async () => {
    mocks.upsert
      .mockResolvedValueOnce({
        ...serializeExecutionMetadata(execution({ status: "completed" })),
        revision: 7,
      })
      .mockResolvedValueOnce({
        ...serializeExecutionMetadata(execution({ status: "completed" })),
        revision: 1,
      });
    await saveRecipeExecution(execution({ id: "shared", status: "completed" }));
    mocks.subjectKey.mockReturnValue("subject:b");
    await saveRecipeExecution(execution({ id: "shared", status: "completed" }));

    expect(mocks.upsert.mock.calls[0][0].revision).toBeUndefined();
    expect(mocks.upsert.mock.calls[1][0].revision).toBeUndefined();
  });

  it("coalesces intermediate snapshots to the latest pending write", async () => {
    vi.useFakeTimers();
    mocks.upsert.mockImplementation(async (input) => ({
      ...input.metadata,
      revision: 1,
    }));
    const writes = [1, 2, 3].map((lastEventId) =>
      saveRecipeExecution(
        execution({ id: "coalesced", status: "running", lastEventId }),
      ),
    );
    expect(mocks.upsert).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(200);
    await Promise.all(writes);
    expect(mocks.upsert).toHaveBeenCalledTimes(1);
    expect(mocks.upsert.mock.calls[0][0].metadata.lastEventId).toBe(3);
    vi.useRealTimers();
  });

  it("flushes a terminal snapshot without waiting for the nonterminal throttle", async () => {
    vi.useFakeTimers();
    mocks.upsert.mockImplementation(async (input) => ({
      ...input.metadata,
      revision: 1,
    }));
    const pending = saveRecipeExecution(
      execution({ id: "terminal-flush", status: "running", lastEventId: 1 }),
    );
    const terminal = saveRecipeExecution(
      execution({ id: "terminal-flush", status: "completed", lastEventId: 2 }),
    );
    await Promise.all([pending, terminal]);
    expect(mocks.upsert).toHaveBeenCalledTimes(1);
    expect(mocks.upsert.mock.calls[0][0].metadata.status).toBe("completed");
    vi.useRealTimers();
  });

  it("hydrates one bounded page plus the server-selected resumable execution", async () => {
    const newest = {
      ...serializeExecutionMetadata(
        execution({ id: "newest", status: "completed" }),
      ),
      revision: 2,
      updatedAt: 20,
    };
    const oldRunning = {
      ...serializeExecutionMetadata(
        execution({ id: "old-running", status: "running" }),
      ),
      revision: 1,
      updatedAt: 10,
    };
    mocks.list.mockResolvedValueOnce({
      executions: [newest],
      nextCursor: "cursor-1",
      resumable: oldRunning,
    });

    await expect(listRecipeExecutions("recipe-1")).resolves.toEqual([
      newest,
      oldRunning,
    ]);
    expect(mocks.list).toHaveBeenCalledWith("recipe-1", {
      cursor: undefined,
      limit: 100,
    });
    expect(mocks.list).toHaveBeenCalledTimes(1);
  });
});
