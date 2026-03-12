// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toastError, toastSuccess } from "@/shared/toast";
import {
  getRecipeJobAnalysis,
  getRecipeJobDataset,
  getRecipeJobStatus,
  streamRecipeJobEvents,
} from "../api";
import type {
  RecipeExecutionKind,
  RecipeExecutionProgress,
  RecipeExecutionRecord,
  RecipeExecutionStatus,
} from "../execution-types";
import {
  DATASET_PAGE_SIZE,
  delay,
  mapJobStatus,
  normalizeAnalysis,
  normalizeDatasetRows,
  toErrorMessage,
} from "./execution-helpers";
import {
  appendExecutionLogLine,
  applyExecutionStatusSnapshot,
  toExecutionLogLine,
} from "./runtime";

type TrackRecipeExecutionParams = {
  label: string;
  kind: RecipeExecutionKind;
  rows: number;
  jobId: string;
  initialExecution: RecipeExecutionRecord;
  notify: boolean;
  onUpsert: (record: RecipeExecutionRecord) => void;
  onSetPreviewErrors: (errors: string[]) => void;
  onPreviewSuccess?: () => void;
};

function isTerminalStatus(status: RecipeExecutionStatus): boolean {
  return status === "completed" || status === "error" || status === "cancelled";
}

function normalizeCompletedProgress(input: {
  latestExecution: RecipeExecutionRecord;
  rows: number;
}): {
  progress: RecipeExecutionProgress;
  columnProgress: RecipeExecutionProgress | null;
} {
  const { latestExecution, rows } = input;
  const progressTotal =
    typeof latestExecution.progress?.total === "number" && latestExecution.progress.total > 0
      ? latestExecution.progress.total
      : latestExecution.rows > 0
        ? latestExecution.rows
        : rows;

  const progress: RecipeExecutionProgress = {
    ...(latestExecution.progress ?? {}),
    done: progressTotal,
    total: progressTotal,
    percent: 100,
    eta_sec: 0,
  };

  const columnProgress =
    latestExecution.column_progress &&
    typeof latestExecution.column_progress.total === "number" &&
    latestExecution.column_progress.total > 0
      ? {
          ...latestExecution.column_progress,
          done: latestExecution.column_progress.total,
          percent: 100,
          eta_sec: 0,
        }
      : latestExecution.column_progress;

  return { progress, columnProgress };
}

export async function trackRecipeExecution({
  label,
  kind,
  rows,
  jobId,
  initialExecution,
  notify,
  onUpsert,
  onSetPreviewErrors,
  onPreviewSuccess,
}: TrackRecipeExecutionParams): Promise<boolean> {
  let done = false;
  let lastStatus: RecipeExecutionStatus = initialExecution.status;
  let completedEventPayload: Record<string, unknown> | null = null;
  let latestExecution: RecipeExecutionRecord = initialExecution;

  const eventsAbortController = new AbortController();
  void streamRecipeJobEvents({
    jobId,
    signal: eventsAbortController.signal,
    lastEventId: latestExecution.lastEventId,
    onEvent: (event) => {
      let changed = false;

      if (typeof event.id === "number") {
        latestExecution = {
          ...latestExecution,
          lastEventId: event.id,
        };
        changed = true;
      }

      const logLine = toExecutionLogLine(event);
      if (logLine) {
        latestExecution = {
          ...latestExecution,
          log_lines: appendExecutionLogLine(latestExecution.log_lines, logLine),
        };
        changed = true;
      }

      const eventType =
        typeof event.payload.type === "string" ? event.payload.type : event.event;

      if (eventType === "job.started") {
        latestExecution = {
          ...latestExecution,
          status: "active",
        };
        onUpsert(latestExecution);
        return;
      }

      if (eventType === "job.completed") {
        lastStatus = "completed";
        completedEventPayload = event.payload;
        done = true;
        latestExecution = {
          ...latestExecution,
          status: "completed",
          finishedAt: Date.now(),
          artifact_path:
            typeof event.payload.artifact_path === "string"
              ? event.payload.artifact_path
              : latestExecution.artifact_path,
          error: null,
        };
        onUpsert(latestExecution);
        return;
      }

      if (eventType === "job.error") {
        lastStatus = "error";
        done = true;
        latestExecution = {
          ...latestExecution,
          status: "error",
          finishedAt: Date.now(),
          error:
            typeof event.payload.error === "string"
              ? event.payload.error
              : latestExecution.error ?? `${label} failed.`,
        };
        onUpsert(latestExecution);
        return;
      }

      if (eventType === "job.cancelling") {
        latestExecution = {
          ...latestExecution,
          status: "cancelling",
        };
        onUpsert(latestExecution);
        return;
      }

      if (changed) {
        onUpsert(latestExecution);
      }
    },
  }).catch(() => {
    // polling is fallback source of truth
  });

  try {
    while (!done) {
      const status = await getRecipeJobStatus(jobId);
      const mappedStatus = mapJobStatus(status.status);
      lastStatus = mappedStatus;
      latestExecution = applyExecutionStatusSnapshot(latestExecution, status);
      onUpsert(latestExecution);

      done = isTerminalStatus(mappedStatus);
      if (!done) {
        await delay(1200);
      }
    }
  } catch (error) {
    const message = toErrorMessage(error, `${label} failed.`);
    latestExecution = {
      ...latestExecution,
      status: "error",
      error: message,
      finishedAt: Date.now(),
    };
    onUpsert(latestExecution);
    if (notify) {
      toastError(`${label} failed`, message);
    }
    return false;
  } finally {
    eventsAbortController.abort();
  }

  if (lastStatus === "completed") {
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        const finalStatus = await getRecipeJobStatus(jobId);
        latestExecution = applyExecutionStatusSnapshot(latestExecution, finalStatus);
      } catch {
        break;
      }
      if (attempt < 2) {
        await delay(250);
      }
    }

    const eventAnalysis = completedEventPayload
      ? completedEventPayload["analysis"]
      : null;
    const eventDataset = completedEventPayload
      ? completedEventPayload["dataset"]
      : null;
    const eventProcessorArtifacts =
      completedEventPayload &&
      typeof completedEventPayload["processor_artifacts"] === "object" &&
      completedEventPayload["processor_artifacts"] !== null
        ? (completedEventPayload["processor_artifacts"] as Record<string, unknown>)
        : null;
    const shouldFetchPreviewDataset = kind === "preview" && !Array.isArray(eventDataset);
    const shouldFetchAnalysis =
      !completedEventPayload ||
      typeof eventAnalysis !== "object" ||
      eventAnalysis === null ||
      kind === "full";

    const [analysisResult, datasetResult] = await Promise.allSettled([
      shouldFetchAnalysis
        ? getRecipeJobAnalysis(jobId)
        : Promise.resolve(eventAnalysis),
      shouldFetchPreviewDataset || kind === "full"
        ? getRecipeJobDataset(jobId, { limit: DATASET_PAGE_SIZE, offset: 0 })
        : Promise.resolve({ dataset: eventDataset ?? [], total: rows }),
    ]);

    const analysis =
      analysisResult.status === "fulfilled"
        ? normalizeAnalysis(analysisResult.value)
        : latestExecution.analysis;
    const datasetResponse =
      datasetResult.status === "fulfilled"
        ? datasetResult.value
        : null;
    const dataset = datasetResponse
      ? normalizeDatasetRows(datasetResponse.dataset)
      : latestExecution.dataset;
    const datasetTotal =
      datasetResponse && typeof datasetResponse.total === "number"
        ? datasetResponse.total
        : latestExecution.datasetTotal;
    const completedProgress = normalizeCompletedProgress({ latestExecution, rows });

    latestExecution = {
      ...latestExecution,
      status: "completed",
      progress: completedProgress.progress,
      column_progress: completedProgress.columnProgress,
      analysis,
      dataset,
      datasetTotal,
      datasetPage: 1,
      datasetPageSize: DATASET_PAGE_SIZE,
      error: null,
      processor_artifacts: eventProcessorArtifacts ?? latestExecution.processor_artifacts,
      finishedAt: latestExecution.finishedAt ?? Date.now(),
    };
    onUpsert(latestExecution);

    if (notify) {
      if (kind === "preview") {
        onSetPreviewErrors([]);
        onPreviewSuccess?.();
        toastSuccess(`Preview generated (${rows} rows).`);
      } else {
        toastSuccess("Full run completed.");
      }
    }
    return true;
  }

  if (lastStatus === "cancelled") {
    latestExecution = {
      ...latestExecution,
      status: "cancelled",
      error: latestExecution.error ?? "Run cancelled.",
      finishedAt: latestExecution.finishedAt ?? Date.now(),
    };
    onUpsert(latestExecution);
    if (notify) {
      toastError(`${label} cancelled`, "The execution was cancelled.");
    }
    return false;
  }

  latestExecution = {
    ...latestExecution,
    status: "error",
    error: latestExecution.error ?? `${label} failed.`,
    finishedAt: latestExecution.finishedAt ?? Date.now(),
  };
  onUpsert(latestExecution);
  if (notify) {
    toastError(`${label} failed`, latestExecution.error ?? "Execution failed.");
  }
  return false;
}
