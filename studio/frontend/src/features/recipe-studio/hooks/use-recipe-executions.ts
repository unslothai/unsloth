import { useCallback, useEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import { toastError } from "@/shared/toast";
import {
  cancelRecipeJob,
  createRecipeJob,
  getRecipeJobDataset,
  validateRecipe,
} from "../api";
import { saveRecipeExecution } from "../data/executions-db";
import type { RecipeExecutionRecord } from "../execution-types";
import {
  DATASET_PAGE_SIZE,
  executionLabel,
  normalizeDatasetRows,
  toErrorMessage,
  withExecutionDefaults,
} from "../executions/execution-helpers";
import {
  findResumableExecution,
  loadSortedRecipeExecutions,
} from "../executions/hydration";
import { createBaseExecutionRecord } from "../executions/runtime";
import { trackRecipeExecution } from "../executions/tracker";
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
  const payloadErrorMessage = payloadResult.errors[0] ?? "Invalid payload.";

  const upsertAndPersist = useCallback(
    (record: RecipeExecutionRecord): void => {
      const normalizedRecord = withExecutionDefaults(record);
      upsertExecution(normalizedRecord);
      void saveRecipeExecution(normalizedRecord).catch((error) => {
        console.error("Save recipe execution failed:", error);
      });
    },
    [upsertExecution],
  );

  useEffect(() => {
    let cancelled = false;

    resetForRecipe();

    async function hydrate(): Promise<void> {
      try {
        const records = await loadSortedRecipeExecutions(recipeId);
        if (cancelled) {
          return;
        }

        setExecutions(records);
        const resumable = findResumableExecution(records);
        if (!resumable?.jobId) {
          return;
        }

        void trackRecipeExecution({
          label: executionLabel(resumable.kind),
          kind: resumable.kind,
          rows: resumable.rows,
          jobId: resumable.jobId,
          initialExecution: resumable,
          notify: false,
          onUpsert: upsertAndPersist,
          onSetPreviewErrors: setPreviewErrors,
          onPreviewSuccess,
        });
      } catch (error) {
        console.error("Load recipe executions failed:", error);
      }
    }

    void hydrate();

    return () => {
      cancelled = true;
    };
  }, [
    onPreviewSuccess,
    recipeId,
    resetForRecipe,
    setExecutions,
    setPreviewErrors,
    upsertAndPersist,
  ]);

  const readPayload = useCallback((): RecipePayload | null => {
    if (payloadResult.errors.length === 0) {
      return payloadResult.payload;
    }
    return null;
  }, [payloadResult.errors.length, payloadResult.payload]);
  const readExecutablePayload = useCallback((): RecipePayload | null => {
    const payload = readPayload();
    if (payload) {
      return payload;
    }

    setPreviewErrors(payloadResult.errors);
    toastError("Invalid recipe payload", payloadErrorMessage);
    return null;
  }, [payloadErrorMessage, payloadResult.errors, readPayload, setPreviewErrors]);

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
      const baseExecution = createBaseExecutionRecord({
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
        const executionWithJob = {
          ...baseExecution,
          jobId: createdJob.job_id,
        };
        upsertAndPersist(executionWithJob);

        return await trackRecipeExecution({
          label,
          kind,
          rows,
          jobId: createdJob.job_id,
          initialExecution: executionWithJob,
          notify: true,
          onUpsert: upsertAndPersist,
          onSetPreviewErrors: setPreviewErrors,
          onPreviewSuccess,
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
      onPreviewSuccess,
      recipeId,
      setFullLoading,
      setPreviewDialogOpen,
      setPreviewErrors,
      setPreviewLoading,
      upsertAndPersist,
    ],
  );

  const runPreview = useCallback(async (): Promise<boolean> => {
    const payload = readExecutablePayload();
    if (!payload) {
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
  }, [previewRows, readExecutablePayload, runExecution, setPreviewErrors]);

  const runFull = useCallback(async (): Promise<boolean> => {
    const payload = readExecutablePayload();
    if (!payload) {
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
  }, [readExecutablePayload, runExecution]);

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
      if (!execution || execution.kind !== "full" || !execution.jobId || page < 1) {
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
