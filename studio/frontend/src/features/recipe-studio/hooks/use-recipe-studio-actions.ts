import { useCallback, useEffect, useMemo, useState } from "react";
import { toastError, toastSuccess } from "@/shared/toast";
import { normalizeNonEmptyName } from "@/utils";
import {
  cancelRecipeJob,
  createRecipeJob,
  getRecipeJobAnalysis,
  getRecipeJobDataset,
  getRecipeJobStatus,
  previewRecipe,
  validateRecipe,
} from "../api";
import { listRecipeExecutions, saveRecipeExecution } from "../data/executions-db";
import type {
  RecipeExecutionAnalysis,
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

const DATASET_PAGE_SIZE = 20;

function buildSignature(name: string, payload: RecipePayload): string {
  return JSON.stringify({ name, payload });
}

function formatSavedLabel(savedAt: number | null): string {
  if (!savedAt) {
    return "Not saved yet";
  }
  const time = new Date(savedAt).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
  return `Saved ${time}`;
}

function toErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error) {
    return error.message;
  }
  return fallback;
}

function normalizeDatasetRows(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(
    (row): row is Record<string, unknown> =>
      typeof row === "object" && row !== null && !Array.isArray(row),
  );
}

function normalizeObject(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function normalizeAnalysis(value: unknown): RecipeExecutionAnalysis | null {
  const normalized = normalizeObject(value);
  if (!normalized) {
    return null;
  }
  return normalized as RecipeExecutionAnalysis;
}

function mapJobStatus(status: string): RecipeExecutionStatus {
  if (status === "active") {
    return "active";
  }
  if (status === "pending") {
    return "pending";
  }
  if (status === "cancelling") {
    return "cancelling";
  }
  if (status === "cancelled") {
    return "cancelled";
  }
  if (status === "completed") {
    return "completed";
  }
  if (status === "error") {
    return "error";
  }
  return "running";
}

function executionSortWeight(status: RecipeExecutionStatus): number {
  if (status === "running" || status === "active" || status === "pending" || status === "cancelling") {
    return 0;
  }
  if (status === "error" || status === "cancelled") {
    return 2;
  }
  return 1;
}

function sortExecutions(records: RecipeExecutionRecord[]): RecipeExecutionRecord[] {
  const next = [...records];
  next.sort((a, b) => {
    const statusDelta = executionSortWeight(a.status) - executionSortWeight(b.status);
    if (statusDelta !== 0) {
      return statusDelta;
    }
    return b.createdAt - a.createdAt;
  });
  return next;
}

function withExecutionDefaults(
  record: RecipeExecutionRecord,
): RecipeExecutionRecord {
  const dataset = Array.isArray(record.dataset) ? record.dataset : [];
  const datasetPageSize =
    typeof record.datasetPageSize === "number" && record.datasetPageSize > 0
      ? record.datasetPageSize
      : DATASET_PAGE_SIZE;
  const datasetPage =
    typeof record.datasetPage === "number" && record.datasetPage > 0
      ? record.datasetPage
      : 1;
  const datasetTotal =
    typeof record.datasetTotal === "number" && record.datasetTotal >= 0
      ? record.datasetTotal
      : dataset.length;

  return {
    ...record,
    dataset,
    datasetTotal,
    datasetPage,
    datasetPageSize,
  };
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function copyTextToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fallthrough to legacy path
  }

  try {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.top = "0";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(textarea);
    return ok;
  } catch {
    return false;
  }
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

  const runPreview = useCallback(async (): Promise<boolean> => {
    const payload = readPayload();
    if (!payload) {
      setPreviewErrors(payloadResult.errors);
      toastError("Invalid recipe payload", payloadErrorMessage);
      return false;
    }
    setPreviewLoading(true);

    const createdAt = Date.now();
    const baseExecution: RecipeExecutionRecord = {
      id: crypto.randomUUID(),
      recipeId,
      jobId: null,
      kind: "preview",
      status: "running",
      rows: previewRows,
      createdAt,
      finishedAt: null,
      recipeSignature: currentSignature,
      stage: "preview",
      current_column: null,
      progress: null,
      model_usage: null,
      lastEventId: null,
      artifact_path: null,
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
    setPreviewDialogOpen(false);

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
        upsertExecution({
          ...baseExecution,
          status: "error",
          error: nextErrors[0],
        });
        setPreviewErrors(nextErrors);
        toastError("Validation failed", nextErrors[0]);
        return false;
      }

      const result = await previewRecipe(previewPayload);
      const dataset = normalizeDatasetRows(result.dataset);
      upsertExecution({
        ...baseExecution,
        status: "completed",
        finishedAt: Date.now(),
        dataset,
        datasetTotal: dataset.length,
        datasetPage: 1,
        datasetPageSize: DATASET_PAGE_SIZE,
        analysis: normalizeAnalysis(result.analysis),
        processor_artifacts: normalizeObject(result.processor_artifacts),
        error: null,
      });
      setPreviewErrors([]);
      toastSuccess(`Preview generated (${previewRows} rows).`);
      onPreviewSuccess?.();
      return true;
    } catch (error) {
      console.error("Preview failed:", error);
      const message = toErrorMessage(error, "Preview request failed.");
      upsertExecution({
        ...baseExecution,
        status: "error",
        finishedAt: Date.now(),
        error: message,
      });
      setPreviewErrors([message]);
      toastError("Preview failed", message);
      return false;
    } finally {
      setPreviewLoading(false);
    }
  }, [
    currentSignature,
    onExecutionStart,
    onPreviewSuccess,
    payloadErrorMessage,
    payloadResult.errors,
    previewRows,
    readPayload,
    recipeId,
    upsertExecution,
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
    const createdAt = Date.now();
    const baseExecution: RecipeExecutionRecord = {
      id: crypto.randomUUID(),
      recipeId,
      jobId: null,
      kind: "full",
      status: "pending",
      rows,
      createdAt,
      finishedAt: null,
      recipeSignature: currentSignature,
      stage: "pending",
      current_column: null,
      progress: null,
      model_usage: null,
      lastEventId: null,
      artifact_path: null,
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
    setFullLoading(true);

    try {
      const fullPayload = {
        ...payload,
        run: {
          ...payload.run,
          rows,
          // biome-ignore lint/style/useNamingConvention: backend schema
          execution_type: "full",
        },
      };
      const createdJob = await createRecipeJob(fullPayload);
      const jobId = createdJob.job_id;
      let done = false;
      let lastStatus: RecipeExecutionStatus = "pending";
      let latestExecution: RecipeExecutionRecord = {
        ...baseExecution,
        jobId,
      };
      upsertExecution(latestExecution);

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

      if (lastStatus === "completed") {
        const [analysisResult, datasetResult] = await Promise.allSettled([
          getRecipeJobAnalysis(jobId),
          getRecipeJobDataset(jobId, { limit: DATASET_PAGE_SIZE, offset: 0 }),
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

        upsertExecution({
          ...latestExecution,
          status: "completed",
          analysis,
          dataset,
          datasetTotal,
          datasetPage: 1,
          datasetPageSize: DATASET_PAGE_SIZE,
          error: null,
          finishedAt: latestExecution.finishedAt ?? Date.now(),
        });
        toastSuccess("Full run completed.");
        return true;
      }

      if (lastStatus === "cancelled") {
        upsertExecution({
          ...latestExecution,
          status: "cancelled",
          error: latestExecution.error ?? "Run cancelled.",
          finishedAt: latestExecution.finishedAt ?? Date.now(),
        });
        toastError("Full run cancelled", "The execution was cancelled.");
        return false;
      }

      upsertExecution({
        ...latestExecution,
        status: "error",
        error: latestExecution.error ?? "Full run failed.",
        finishedAt: latestExecution.finishedAt ?? Date.now(),
      });
      toastError("Full run failed", latestExecution.error ?? "Execution failed.");
      return false;
    } catch (error) {
      const message = toErrorMessage(error, "Full run request failed.");
      upsertExecution({
        ...baseExecution,
        status: "error",
        error: message,
        finishedAt: Date.now(),
      });
      toastError("Full run failed", message);
      return false;
    } finally {
      setFullLoading(false);
    }
  }, [
    currentSignature,
    onExecutionStart,
    payloadErrorMessage,
    payloadResult.errors,
    readPayload,
    recipeId,
    upsertExecution,
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
