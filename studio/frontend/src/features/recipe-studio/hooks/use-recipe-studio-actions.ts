import { useCallback, useEffect, useMemo, useState } from "react";
import { toastError, toastSuccess } from "@/shared/toast";
import { normalizeNonEmptyName } from "@/utils";
import {
  cancelRecipeJob,
  createRecipeJob,
  getRecipeJobAnalysis,
  getRecipeJobDataset,
  getRecipeJobStatus,
  streamRecipeJobEvents,
  type JobEvent,
  validateRecipe,
} from "../api";
import { listRecipeExecutions, saveRecipeExecution } from "../data/executions-db";
import {
  buildSignature,
  copyTextToClipboard,
  DATASET_PAGE_SIZE,
  delay,
  executionLabel,
  formatSavedLabel,
  mapJobStatus,
  normalizeAnalysis,
  normalizeDatasetRows,
  normalizeObject,
  sortExecutions,
  toErrorMessage,
  withExecutionDefaults,
} from "../executions/execution-helpers";
import type {
  RecipeExecutionRecord,
  RecipeExecutionStatus,
} from "../execution-types";
import { importRecipePayload, type RecipeSnapshot } from "../utils/import";
import type { RecipePayload, RecipePayloadResult } from "../utils/payload/types";

type SaveTone = "success" | "error";

type PersistRecipeFn = (input: {
  id: string | null;
  name: string;
  payload: RecipePayload;
}) => Promise<{
  id: string;
  updatedAt: number;
}>;

type UseRecipeStudioActionsParams = {
  recipeId: string;
  initialRecipeName: string;
  initialPayload: RecipePayload;
  initialSavedAt: number;
  payloadResult: RecipePayloadResult;
  onPersistRecipe: PersistRecipeFn;
  resetRecipe: () => void;
  loadRecipe: (snapshot: RecipeSnapshot) => void;
  getCurrentPayloadFromStore: () => RecipePayload;
  onExecutionStart?: () => void;
  onPreviewSuccess?: () => void;
};

type UseRecipeStudioActionsResult = {
  workflowName: string;
  setWorkflowName: (value: string) => void;
  saveLoading: boolean;
  saveTone: SaveTone;
  savedAtLabel: string;
  copied: boolean;
  importOpen: boolean;
  setImportOpen: (open: boolean) => void;
  previewDialogOpen: boolean;
  setPreviewDialogOpen: (open: boolean) => void;
  previewRows: number;
  setPreviewRows: (rows: number) => void;
  previewErrors: string[];
  previewLoading: boolean;
  fullLoading: boolean;
  currentSignature: string;
  executions: RecipeExecutionRecord[];
  selectedExecutionId: string | null;
  setSelectedExecutionId: (id: string) => void;
  persistRecipe: () => Promise<void>;
  openPreviewDialog: () => void;
  runPreview: () => Promise<boolean>;
  runFull: () => Promise<boolean>;
  cancelExecution: (id: string) => Promise<void>;
  loadExecutionDatasetPage: (id: string, page: number) => Promise<void>;
  copyRecipe: () => Promise<void>;
  importRecipe: (value: string) => string | null;
};

type JobCompletedEventPayload = {
  analysis?: unknown;
  dataset?: unknown;
  // biome-ignore lint/style/useNamingConvention: backend schema
  artifact_path?: unknown;
  error?: unknown;
  type?: unknown;
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

export function useRecipeStudioActions({
  recipeId,
  initialRecipeName,
  initialPayload,
  initialSavedAt,
  payloadResult,
  onPersistRecipe,
  resetRecipe,
  loadRecipe,
  getCurrentPayloadFromStore,
  onExecutionStart,
  onPreviewSuccess,
}: UseRecipeStudioActionsParams): UseRecipeStudioActionsResult {
  const [workflowName, setWorkflowName] = useState("Unnamed");
  const [lastSavedAt, setLastSavedAt] = useState<number | null>(null);
  const [savedSignature, setSavedSignature] = useState<string>("");
  const [saveLoading, setSaveLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false);
  const [previewRows, setPreviewRows] = useState(5);
  const [previewErrors, setPreviewErrors] = useState<string[]>([]);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [fullLoading, setFullLoading] = useState(false);
  const [executions, setExecutions] = useState<RecipeExecutionRecord[]>([]);
  const [selectedExecutionId, setSelectedExecutionId] = useState<string | null>(
    null,
  );

  const normalizedWorkflowName = useMemo(
    () => normalizeNonEmptyName(workflowName, "Unnamed"),
    [workflowName],
  );
  const currentPayload = payloadResult.payload;
  const currentSignature = useMemo(
    () => buildSignature(normalizedWorkflowName, currentPayload),
    [currentPayload, normalizedWorkflowName],
  );
  const isDirty = savedSignature.length > 0 && currentSignature !== savedSignature;
  const saveTone: SaveTone = !isDirty && Boolean(lastSavedAt) ? "success" : "error";
  const savedAtLabel = formatSavedLabel(lastSavedAt);
  const payloadErrorMessage = payloadResult.errors[0] ?? "Invalid payload.";

  useEffect(() => {
    const nextName = normalizeNonEmptyName(initialRecipeName, "Unnamed");
    resetRecipe();
    setWorkflowName(nextName);
    setLastSavedAt(initialSavedAt);
    setCopied(false);
    setPreviewErrors([]);
    setPreviewDialogOpen(false);
    setSelectedExecutionId(null);

    const parsed = importRecipePayload(JSON.stringify(initialPayload));
    if (parsed.snapshot) {
      loadRecipe(parsed.snapshot);
    } else {
      console.error("Failed to load recipe payload.", parsed.errors);
    }

    const payload = getCurrentPayloadFromStore();
    setSavedSignature(buildSignature(nextName, payload));
  }, [
    getCurrentPayloadFromStore,
    initialPayload,
    initialRecipeName,
    initialSavedAt,
    loadRecipe,
    recipeId,
    resetRecipe,
  ]);

  useEffect(() => {
    let cancelled = false;
    setExecutions([]);
    setSelectedExecutionId(null);

    async function loadExecutions(): Promise<void> {
      try {
        const records = await listRecipeExecutions(recipeId);
        if (cancelled) {
          return;
        }
        const sortedRecords = sortExecutions(records.map(withExecutionDefaults));
        setExecutions(sortedRecords);
        setSelectedExecutionId(sortedRecords[0]?.id ?? null);
      } catch (error) {
        console.error("Load recipe executions failed:", error);
      }
    }

    void loadExecutions();

    return () => {
      cancelled = true;
    };
  }, [recipeId]);

  const upsertExecution = useCallback((record: RecipeExecutionRecord): void => {
    const normalizedRecord = withExecutionDefaults(record);
    setExecutions((current) => {
      const withoutCurrent = current.filter((item) => item.id !== normalizedRecord.id);
      return sortExecutions([normalizedRecord, ...withoutCurrent]);
    });
    setSelectedExecutionId(normalizedRecord.id);
    void saveRecipeExecution(normalizedRecord).catch((error) => {
      console.error("Save recipe execution failed:", error);
    });
  }, []);

  const persistRecipe = useCallback(async (): Promise<void> => {
    if (saveLoading) {
      return;
    }
    const nextName = normalizeNonEmptyName(workflowName, "Unnamed");
    if (nextName !== workflowName) {
      setWorkflowName(nextName);
    }
    setSaveLoading(true);
    try {
      const result = await onPersistRecipe({
        id: recipeId,
        name: nextName,
        payload: currentPayload,
      });
      setLastSavedAt(result.updatedAt);
      setSavedSignature(buildSignature(nextName, currentPayload));
    } catch (error) {
      console.error("Save recipe failed:", error);
      toastError("Save failed", "Could not save recipe.");
    } finally {
      setSaveLoading(false);
    }
  }, [currentPayload, onPersistRecipe, recipeId, saveLoading, workflowName]);

  useEffect(() => {
    if (!isDirty || saveLoading) {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      void persistRecipe();
    }, 800);
    return () => window.clearTimeout(timeoutId);
  }, [isDirty, persistRecipe, saveLoading]);

  const readPayload = useCallback((): RecipePayload | null => {
    if (payloadResult.errors.length === 0) {
      return payloadResult.payload;
    }
    return null;
  }, [payloadResult.errors.length, payloadResult.payload]);

  function openPreviewDialog(): void {
    setPreviewErrors([]);
    setPreviewDialogOpen(true);
  }

  const runJobExecution = useCallback(async (input: {
    kind: "preview" | "full";
    payload: RecipePayload;
    rows: number;
  }): Promise<boolean> => {
    const { kind, payload, rows } = input;
    const setLoading = kind === "preview" ? setPreviewLoading : setFullLoading;
    const label = executionLabel(kind);

    setLoading(true);
    const createdAt = Date.now();
    const baseExecution: RecipeExecutionRecord = {
      id: crypto.randomUUID(),
      recipeId,
      jobId: null,
      kind,
      status: "pending",
      rows,
      createdAt,
      finishedAt: null,
      recipeSignature: currentSignature,
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

    upsertExecution(baseExecution);
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
      const jobId = createdJob.job_id;
      let done = false;
      let lastStatus: RecipeExecutionStatus = "pending";
      let completedEventPayload: JobCompletedEventPayload | null = null;
      let latestExecution: RecipeExecutionRecord = {
        ...baseExecution,
        jobId,
      };
      upsertExecution(latestExecution);

      const eventsAbortController = new AbortController();
      void streamRecipeJobEvents({
        jobId,
        signal: eventsAbortController.signal,
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
            upsertExecution(latestExecution);
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
            upsertExecution(latestExecution);
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
            upsertExecution(latestExecution);
            return;
          }

          if (eventType === "job.cancelling") {
            latestExecution = {
              ...latestExecution,
              status: "cancelling",
            };
            upsertExecution(latestExecution);
            return;
          }

          if (changed) {
            upsertExecution(latestExecution);
          }
        },
      }).catch(() => {
        // polling remains fallback source of truth
      });

      try {
        while (!done) {
          const status = await getRecipeJobStatus(jobId);
          const mappedStatus = mapJobStatus(status.status);
          lastStatus = mappedStatus;

          latestExecution = {
            ...latestExecution,
            status: mappedStatus,
            rows: status.rows ?? latestExecution.rows,
            stage: status.stage ?? latestExecution.stage,
            current_column: status.current_column ?? null,
            progress: (normalizeObject(status.progress) as RecipeExecutionRecord["progress"]) ?? null,
            column_progress:
              (normalizeObject(status.column_progress) as RecipeExecutionRecord["column_progress"]) ?? null,
            model_usage: normalizeObject(status.model_usage),
            artifact_path: status.artifact_path ?? latestExecution.artifact_path,
            error: status.error ?? null,
            finishedAt:
              mappedStatus === "completed" ||
              mappedStatus === "error" ||
              mappedStatus === "cancelled"
                ? Date.now()
                : null,
          };
          upsertExecution(latestExecution);

          done =
            mappedStatus === "completed" ||
            mappedStatus === "error" ||
            mappedStatus === "cancelled";
          if (!done) {
            await delay(1200);
          }
        }
      } finally {
        eventsAbortController.abort();
      }

      if (lastStatus === "completed") {
        for (let attempt = 0; attempt < 3; attempt += 1) {
          try {
            const finalStatus = await getRecipeJobStatus(jobId);
            latestExecution = {
              ...latestExecution,
              status: mapJobStatus(finalStatus.status),
              rows: finalStatus.rows ?? latestExecution.rows,
              stage: finalStatus.stage ?? latestExecution.stage,
              current_column: finalStatus.current_column ?? latestExecution.current_column,
              progress:
                (normalizeObject(finalStatus.progress) as RecipeExecutionRecord["progress"]) ??
                latestExecution.progress,
              column_progress:
                (normalizeObject(
                  finalStatus.column_progress,
                ) as RecipeExecutionRecord["column_progress"]) ??
                latestExecution.column_progress,
              model_usage: normalizeObject(finalStatus.model_usage) ?? latestExecution.model_usage,
              artifact_path: finalStatus.artifact_path ?? latestExecution.artifact_path,
              error: finalStatus.error ?? latestExecution.error,
              finishedAt: latestExecution.finishedAt ?? Date.now(),
            };
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

        upsertExecution({
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
        });

        if (kind === "preview") {
          setPreviewErrors([]);
          onPreviewSuccess?.();
          toastSuccess(`Preview generated (${rows} rows).`);
        } else {
          toastSuccess("Full run completed.");
        }
        return true;
      }

      if (lastStatus === "cancelled") {
        upsertExecution({
          ...latestExecution,
          status: "cancelled",
          error: latestExecution.error ?? "Run cancelled.",
          finishedAt: latestExecution.finishedAt ?? Date.now(),
        });
        toastError(`${label} cancelled`, "The execution was cancelled.");
        return false;
      }

      upsertExecution({
        ...latestExecution,
        status: "error",
        error: latestExecution.error ?? `${label} failed.`,
        finishedAt: latestExecution.finishedAt ?? Date.now(),
      });
      toastError(`${label} failed`, latestExecution.error ?? "Execution failed.");
      return false;
    } catch (error) {
      const message = toErrorMessage(error, `${label} request failed.`);
      upsertExecution({
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
  }, [
    currentSignature,
    onExecutionStart,
    onPreviewSuccess,
    recipeId,
    upsertExecution,
  ]);

  const runPreview = useCallback(async (): Promise<boolean> => {
    const payload = readPayload();
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

    return runJobExecution({
      kind: "preview",
      payload,
      rows: previewRows,
    });
  }, [
    payloadErrorMessage,
    payloadResult.errors,
    previewRows,
    readPayload,
    runJobExecution,
  ]);

  const runFull = useCallback(async (): Promise<boolean> => {
    const payload = readPayload();
    if (!payload) {
      setPreviewErrors(payloadResult.errors);
      toastError("Invalid recipe payload", payloadErrorMessage);
      return false;
    }

    const requestedRows = Number(payload.run?.rows);
    const rows = Number.isFinite(requestedRows) && requestedRows > 0
      ? Math.floor(requestedRows)
      : 1000;

    return runJobExecution({
      kind: "full",
      payload,
      rows,
    });
  }, [
    payloadErrorMessage,
    payloadResult.errors,
    readPayload,
    runJobExecution,
  ]);

  const cancelExecution = useCallback(async (id: string): Promise<void> => {
    const execution = executions.find((entry) => entry.id === id);
    if (!execution?.jobId) {
      return;
    }
    try {
      await cancelRecipeJob(execution.jobId);
      upsertExecution({
        ...execution,
        status: "cancelling",
      });
    } catch (error) {
      const message = toErrorMessage(error, "Could not cancel execution.");
      toastError("Cancel failed", message);
    }
  }, [executions, upsertExecution]);

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
        upsertExecution({
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
    [executions, upsertExecution],
  );

  const selectExecution = useCallback((id: string): void => {
    setSelectedExecutionId(id);
  }, []);

  const copyRecipe = useCallback(async (): Promise<void> => {
    setCopied(false);
    const payload = readPayload();
    if (!payload) {
      toastError("Copy failed", payloadErrorMessage);
      return;
    }
    try {
      const ok = await copyTextToClipboard(JSON.stringify(payload, null, 2));
      if (!ok) {
        throw new Error("Clipboard not available.");
      }
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
      toastSuccess("Payload copied");
    } catch (error) {
      console.error("Copy failed:", error);
      toastError("Copy failed", "Could not copy payload.");
    }
  }, [payloadErrorMessage, readPayload]);

  const importRecipe = useCallback(
    (value: string): string | null => {
      const result = importRecipePayload(value);
      if (result.errors.length > 0 || !result.snapshot) {
        return result.errors[0] ?? "Invalid payload.";
      }
      loadRecipe(result.snapshot);
      toastSuccess("Recipe imported");
      return null;
    },
    [loadRecipe],
  );

  return {
    workflowName,
    setWorkflowName,
    saveLoading,
    saveTone,
    savedAtLabel,
    copied,
    importOpen,
    setImportOpen,
    previewDialogOpen,
    setPreviewDialogOpen,
    previewRows,
    setPreviewRows,
    previewErrors,
    previewLoading,
    fullLoading,
    currentSignature,
    executions,
    selectedExecutionId,
    setSelectedExecutionId: selectExecution,
    persistRecipe,
    openPreviewDialog,
    runPreview,
    runFull,
    cancelExecution,
    loadExecutionDatasetPage,
    copyRecipe,
    importRecipe,
  };
}
