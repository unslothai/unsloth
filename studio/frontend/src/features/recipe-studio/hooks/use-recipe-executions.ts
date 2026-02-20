import { useCallback, useEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import { toastError, toastSuccess } from "@/shared/toast";
import {
  cancelRecipeJob,
  createRecipeJob,
  getRecipeJobAnalysis,
  getRecipeJobDataset,
  getRecipeJobStatus,
  streamRecipeJobEvents,
  validateRecipe,
  type JobEvent,
  type JobStatusResponse,
} from "../api";
import { listRecipeExecutions, saveRecipeExecution } from "../data/executions-db";
import type {
  RecipeExecutionRecord,
  RecipeExecutionStatus,
} from "../execution-types";
import {
  DATASET_PAGE_SIZE,
  delay,
  executionLabel,
  isExecutionInProgress,
  mapJobStatus,
  normalizeAnalysis,
  normalizeDatasetRows,
  normalizeObject,
  sortExecutions,
  toErrorMessage,
  withExecutionDefaults,
} from "../executions/execution-helpers";
import { useRecipeExecutionsStore } from "../stores/recipe-executions";
import type { RecipePayload, RecipePayloadResult } from "../utils/payload/types";

type UseRecipeExecutionsParams = {
  recipeId: string;
  currentSignature: string;
  payloadResult: RecipePayloadResult;
  onExecutionStart?: () => void;
  onPreviewSuccess?: () => void;
};

type UseRecipeExecutionsResult = {
  previewDialogOpen: boolean;
  setPreviewDialogOpen: (open: boolean) => void;
  previewRows: number;
  setPreviewRows: (rows: number) => void;
  previewErrors: string[];
  previewLoading: boolean;
  fullLoading: boolean;
  executions: RecipeExecutionRecord[];
  selectedExecutionId: string | null;
  setSelectedExecutionId: (id: string) => void;
  openPreviewDialog: () => void;
  runPreview: () => Promise<boolean>;
  runFull: () => Promise<boolean>;
  cancelExecution: (id: string) => Promise<void>;
  loadExecutionDatasetPage: (id: string, page: number) => Promise<void>;
};

const MAX_LOG_LINES = 1500;

function formatEventTime(ts: unknown): string {
  if (typeof ts !== "number" || !Number.isFinite(ts)) {
    return new Date().toLocaleTimeString();
  }
  const ms = ts > 10_000_000_000 ? ts : ts * 1000;
  return new Date(ms).toLocaleTimeString();
}

function appendLogLine(lines: string[], nextLine: string): string[] {
  const next = [...lines, nextLine];
  if (next.length <= MAX_LOG_LINES) {
    return next;
  }
  return next.slice(next.length - MAX_LOG_LINES);
}

function toLogLine(event: JobEvent): string | null {
  const eventType =
    typeof event.payload.type === "string" ? event.payload.type : event.event;
  const ts = formatEventTime(event.payload.ts);

  if (eventType === "log") {
    const message =
      typeof event.payload.message === "string" ? event.payload.message.trim() : "";
    if (!message) {
      return null;
    }
    const level =
      typeof event.payload.level === "string" && event.payload.level.length > 0
        ? event.payload.level.toUpperCase()
        : "INFO";
    return `[${ts}] [${level}] ${message}`;
  }

  if (eventType === "job.started") {
    return `[${ts}] [INFO] Job started`;
  }
  if (eventType === "job.completed") {
    return `[${ts}] [INFO] Job completed`;
  }
  if (eventType === "job.cancelling") {
    return `[${ts}] [WARN] Cancellation requested`;
  }
  if (eventType === "job.cancelled") {
    return `[${ts}] [WARN] Job cancelled`;
  }
  if (eventType === "job.error") {
    const error =
      typeof event.payload.error === "string" && event.payload.error.length > 0
        ? event.payload.error
        : "Job failed";
    return `[${ts}] [ERROR] ${error}`;
  }

  return null;
}

function applyStatusSnapshot(
  execution: RecipeExecutionRecord,
  status: JobStatusResponse,
): RecipeExecutionRecord {
  const mappedStatus = mapJobStatus(status.status);
  return {
    ...execution,
    status: mappedStatus,
    rows: status.rows ?? execution.rows,
    stage: status.stage ?? execution.stage,
    current_column: status.current_column ?? null,
    progress: (normalizeObject(status.progress) as RecipeExecutionRecord["progress"]) ?? null,
    column_progress:
      (normalizeObject(status.column_progress) as RecipeExecutionRecord["column_progress"]) ??
      null,
    model_usage: normalizeObject(status.model_usage),
    artifact_path: status.artifact_path ?? execution.artifact_path,
    error: status.error ?? null,
    finishedAt:
      mappedStatus === "completed" ||
      mappedStatus === "error" ||
      mappedStatus === "cancelled"
        ? Date.now()
        : null,
  };
}

function createBaseExecution(input: {
  recipeId: string;
  kind: "preview" | "full";
  rows: number;
  currentSignature: string;
}): RecipeExecutionRecord {
  const createdAt = Date.now();
  return {
    id: crypto.randomUUID(),
    recipeId: input.recipeId,
    jobId: null,
    kind: input.kind,
    status: "pending",
    rows: input.rows,
    createdAt,
    finishedAt: null,
    recipeSignature: input.currentSignature,
    stage: "pending",
    current_column: null,
    progress: null,
    column_progress: null,
    model_usage: null,
    lastEventId: null,
    artifact_path: null,
    log_lines: [],
    dataset: [],
    datasetTotal: 0,
    datasetPage: 1,
    datasetPageSize: DATASET_PAGE_SIZE,
    analysis: null,
    processor_artifacts: null,
    error: null,
  };
}

export function useRecipeExecutions({
  recipeId,
  currentSignature,
  payloadResult,
  onExecutionStart,
  onPreviewSuccess,
}: UseRecipeExecutionsParams): UseRecipeExecutionsResult {
  const {
    previewDialogOpen,
    previewRows,
    previewErrors,
    previewLoading,
    fullLoading,
    executions,
    selectedExecutionId,
    setPreviewDialogOpen,
    setPreviewRows,
    setPreviewErrors,
    setPreviewLoading,
    setFullLoading,
    setExecutions,
    upsertExecution,
    selectExecution,
    resetForRecipe,
  } = useRecipeExecutionsStore(
    useShallow((state) => ({
      previewDialogOpen: state.previewDialogOpen,
      previewRows: state.previewRows,
      previewErrors: state.previewErrors,
      previewLoading: state.previewLoading,
      fullLoading: state.fullLoading,
      executions: state.executions,
      selectedExecutionId: state.selectedExecutionId,
      setPreviewDialogOpen: state.setPreviewDialogOpen,
      setPreviewRows: state.setPreviewRows,
      setPreviewErrors: state.setPreviewErrors,
      setPreviewLoading: state.setPreviewLoading,
      setFullLoading: state.setFullLoading,
      setExecutions: state.setExecutions,
      upsertExecution: state.upsertExecution,
      selectExecution: state.selectExecution,
      resetForRecipe: state.resetForRecipe,
    })),
  );

  const upsertAndPersist = useCallback(
    (record: RecipeExecutionRecord): void => {
      const normalized = withExecutionDefaults(record);
      upsertExecution(normalized);
      void saveRecipeExecution(normalized).catch((error) => {
        console.error("Save recipe execution failed:", error);
      });
    },
    [upsertExecution],
  );

  const trackExecution = useCallback(
    async (input: {
      label: string;
      kind: "preview" | "full";
      rows: number;
      jobId: string;
      initialExecution: RecipeExecutionRecord;
      notify: boolean;
    }): Promise<boolean> => {
      const { label, kind, rows, jobId, notify } = input;
      let done = false;
      let lastStatus: RecipeExecutionStatus = input.initialExecution.status;
      let completedEventPayload: Record<string, unknown> | null = null;
      let latestExecution: RecipeExecutionRecord = input.initialExecution;

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

          const logLine = toLogLine(event);
          if (logLine) {
            latestExecution = {
              ...latestExecution,
              log_lines: appendLogLine(latestExecution.log_lines, logLine),
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
            upsertAndPersist(latestExecution);
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
            upsertAndPersist(latestExecution);
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
            upsertAndPersist(latestExecution);
            return;
          }

          if (eventType === "job.cancelling") {
            latestExecution = {
              ...latestExecution,
              status: "cancelling",
            };
            upsertAndPersist(latestExecution);
            return;
          }

          if (changed) {
            upsertAndPersist(latestExecution);
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
          latestExecution = applyStatusSnapshot(latestExecution, status);
          upsertAndPersist(latestExecution);

          done =
            mappedStatus === "completed" ||
            mappedStatus === "error" ||
            mappedStatus === "cancelled";
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
        upsertAndPersist(latestExecution);
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
            latestExecution = applyStatusSnapshot(latestExecution, finalStatus);
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
        const shouldFetchPreviewDataset =
          kind === "preview" && !Array.isArray(eventDataset);
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

        const progressTotal =
          typeof latestExecution.progress?.total === "number" &&
          latestExecution.progress.total > 0
            ? latestExecution.progress.total
            : latestExecution.rows > 0
              ? latestExecution.rows
              : rows;

        const completedProgress: RecipeExecutionRecord["progress"] = {
          ...(latestExecution.progress ?? {}),
          done: progressTotal,
          total: progressTotal,
          percent: 100,
          eta_sec: 0,
        };
        const completedColumnProgress: RecipeExecutionRecord["column_progress"] =
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

        latestExecution = {
          ...latestExecution,
          status: "completed",
          progress: completedProgress,
          column_progress: completedColumnProgress,
          analysis,
          dataset,
          datasetTotal,
          datasetPage: 1,
          datasetPageSize: DATASET_PAGE_SIZE,
          error: null,
          finishedAt: latestExecution.finishedAt ?? Date.now(),
        };
        upsertAndPersist(latestExecution);

        if (notify) {
          if (kind === "preview") {
            setPreviewErrors([]);
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
        upsertAndPersist(latestExecution);
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
      upsertAndPersist(latestExecution);
      if (notify) {
        toastError(`${label} failed`, latestExecution.error ?? "Execution failed.");
      }
      return false;
    },
    [onPreviewSuccess, setPreviewErrors, upsertAndPersist],
  );

  useEffect(() => {
    let cancelled = false;

    resetForRecipe();

    async function loadExecutions(): Promise<void> {
      try {
        const records = await listRecipeExecutions(recipeId);
        if (cancelled) {
          return;
        }
        const sortedRecords = sortExecutions(records.map(withExecutionDefaults));
        setExecutions(sortedRecords);

        const activeExecution = sortedRecords.find(
          (record) => record.jobId && isExecutionInProgress(record.status),
        );

        if (activeExecution?.jobId) {
          void trackExecution({
            label: executionLabel(activeExecution.kind),
            kind: activeExecution.kind,
            rows: activeExecution.rows,
            jobId: activeExecution.jobId,
            initialExecution: activeExecution,
            notify: false,
          });
        }
      } catch (error) {
        console.error("Load recipe executions failed:", error);
      }
    }

    void loadExecutions();

    return () => {
      cancelled = true;
    };
  }, [recipeId, resetForRecipe, setExecutions, trackExecution]);

  const readPayload = useCallback((): RecipePayload | null => {
    if (payloadResult.errors.length === 0) {
      return payloadResult.payload;
    }
    return null;
  }, [payloadResult.errors.length, payloadResult.payload]);

  const openPreviewDialog = useCallback((): void => {
    setPreviewErrors([]);
    setPreviewDialogOpen(true);
  }, [setPreviewDialogOpen, setPreviewErrors]);

  const runExecution = useCallback(
    async (input: {
      kind: "preview" | "full";
      payload: RecipePayload;
      rows: number;
    }): Promise<boolean> => {
      const { kind, payload, rows } = input;
      const setLoading = kind === "preview" ? setPreviewLoading : setFullLoading;
      const label = executionLabel(kind);

      setLoading(true);
      const baseExecution = createBaseExecution({
        recipeId,
        kind,
        rows,
        currentSignature,
      });

      upsertAndPersist(baseExecution);
      onExecutionStart?.();
      if (kind === "preview") {
        setPreviewDialogOpen(false);
      }

      try {
        const jobPayload = {
          ...payload,
          run: {
            ...payload.run,
            rows,
            // biome-ignore lint/style/useNamingConvention: backend schema
            execution_type: kind,
          },
        };
        const createdJob = await createRecipeJob(jobPayload);
        const latestExecution = {
          ...baseExecution,
          jobId: createdJob.job_id,
        };
        upsertAndPersist(latestExecution);

        return await trackExecution({
          label,
          kind,
          rows,
          jobId: createdJob.job_id,
          initialExecution: latestExecution,
          notify: true,
        });
      } catch (error) {
        const message = toErrorMessage(error, `${label} request failed.`);
        upsertAndPersist({
          ...baseExecution,
          status: "error",
          error: message,
          finishedAt: Date.now(),
        });
        if (kind === "preview") {
          setPreviewErrors([message]);
        }
        toastError(`${label} failed`, message);
        return false;
      } finally {
        setLoading(false);
      }
    },
    [
      currentSignature,
      onExecutionStart,
      recipeId,
      setFullLoading,
      setPreviewDialogOpen,
      setPreviewErrors,
      setPreviewLoading,
      trackExecution,
      upsertAndPersist,
    ],
  );

  const runPreview = useCallback(async (): Promise<boolean> => {
    const payload = readPayload();
    const payloadErrorMessage = payloadResult.errors[0] ?? "Invalid payload.";
    if (!payload) {
      setPreviewErrors(payloadResult.errors);
      toastError("Invalid recipe payload", payloadErrorMessage);
      return false;
    }

    const previewPayload = {
      ...payload,
      run: {
        ...payload.run,
        rows: previewRows,
      },
    };
    try {
      const validation = await validateRecipe(previewPayload);
      if (!validation.valid) {
        const errors = validation.errors.map((item) => item.message);
        const fallback = validation.raw_detail ?? "Validation failed.";
        const nextErrors = errors.length > 0 ? errors : [fallback];
        setPreviewErrors(nextErrors);
        toastError("Validation failed", nextErrors[0]);
        return false;
      }
    } catch (error) {
      const message = toErrorMessage(error, "Validation failed.");
      setPreviewErrors([message]);
      toastError("Validation failed", message);
      return false;
    }

    return runExecution({
      kind: "preview",
      payload,
      rows: previewRows,
    });
  }, [payloadResult.errors, previewRows, readPayload, runExecution, setPreviewErrors]);

  const runFull = useCallback(async (): Promise<boolean> => {
    const payload = readPayload();
    const payloadErrorMessage = payloadResult.errors[0] ?? "Invalid payload.";
    if (!payload) {
      setPreviewErrors(payloadResult.errors);
      toastError("Invalid recipe payload", payloadErrorMessage);
      return false;
    }

    const requestedRows = Number(payload.run?.rows);
    const rows =
      Number.isFinite(requestedRows) && requestedRows > 0
        ? Math.floor(requestedRows)
        : 1000;

    return runExecution({
      kind: "full",
      payload,
      rows,
    });
  }, [payloadResult.errors, readPayload, runExecution, setPreviewErrors]);

  const cancelExecution = useCallback(
    async (id: string): Promise<void> => {
      const execution = executions.find((entry) => entry.id === id);
      if (!execution?.jobId) {
        return;
      }
      try {
        await cancelRecipeJob(execution.jobId);
        upsertAndPersist({
          ...execution,
          status: "cancelling",
        });
      } catch (error) {
        const message = toErrorMessage(error, "Could not cancel execution.");
        toastError("Cancel failed", message);
      }
    },
    [executions, upsertAndPersist],
  );

  const loadExecutionDatasetPage = useCallback(
    async (id: string, page: number): Promise<void> => {
      const execution = executions.find((entry) => entry.id === id);
      if (!execution || execution.kind !== "full" || !execution.jobId) {
        return;
      }
      if (page < 1) {
        return;
      }

      const pageSize = execution.datasetPageSize || DATASET_PAGE_SIZE;
      const offset = (page - 1) * pageSize;
      try {
        const response = await getRecipeJobDataset(execution.jobId, {
          limit: pageSize,
          offset,
        });
        const dataset = normalizeDatasetRows(response.dataset);
        const total =
          typeof response.total === "number" ? response.total : execution.datasetTotal;
        upsertAndPersist({
          ...execution,
          dataset,
          datasetTotal: total,
          datasetPage: page,
        });
      } catch (error) {
        const message = toErrorMessage(error, "Could not load dataset page.");
        toastError("Dataset page failed", message);
      }
    },
    [executions, upsertAndPersist],
  );

  const setSelectedExecutionId = useCallback(
    (id: string): void => {
      selectExecution(id);
    },
    [selectExecution],
  );

  return {
    previewDialogOpen,
    setPreviewDialogOpen,
    previewRows,
    setPreviewRows,
    previewErrors,
    previewLoading,
    fullLoading,
    executions,
    selectedExecutionId,
    setSelectedExecutionId,
    openPreviewDialog,
    runPreview,
    runFull,
    cancelExecution,
    loadExecutionDatasetPage,
  };
}
