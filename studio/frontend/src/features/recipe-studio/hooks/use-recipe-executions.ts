import { useCallback, useEffect, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { toastError } from "@/shared/toast";
import {
  cancelRecipeJob,
  createRecipeJob,
  getRecipeJobDataset,
  validateRecipe,
} from "../api";
import { saveRecipeExecution } from "../data/executions-db";
import type {
  RecipeExecutionKind,
  RecipeExecutionRecord,
} from "../execution-types";
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
import {
  buildExecutionPayload,
  sanitizeExecutionRows,
} from "../executions/run-settings";
import { trackRecipeExecution } from "../executions/tracker";
import {
  type RecipeRunSettings,
  useRecipeExecutionsStore,
} from "../stores/recipe-executions";
import type { RecipePayload, RecipePayloadResult } from "../utils/payload/types";

type UseRecipeExecutionsParams = {
  recipeId: string;
  currentSignature: string;
  payloadResult: RecipePayloadResult;
  onExecutionStart?: () => void;
  onPreviewSuccess?: () => void;
};

type UseRecipeExecutionsResult = {
  runDialogOpen: boolean;
  runDialogKind: RecipeExecutionKind;
  setRunDialogKind: (kind: RecipeExecutionKind) => void;
  setRunDialogOpen: (open: boolean) => void;
  previewRows: number;
  fullRows: number;
  setPreviewRows: (rows: number) => void;
  setFullRows: (rows: number) => void;
  runErrors: string[];
  runSettings: RecipeRunSettings;
  setRunSettings: (patch: Partial<RecipeRunSettings>) => void;
  previewLoading: boolean;
  fullLoading: boolean;
  executions: RecipeExecutionRecord[];
  selectedExecutionId: string | null;
  setSelectedExecutionId: (id: string) => void;
  openRunDialog: (kind: RecipeExecutionKind) => void;
  runFromDialog: () => Promise<boolean>;
  validateFromDialog: () => Promise<boolean>;
  validateLoading: boolean;
  validateResult: {
    valid: boolean;
    errors: string[];
    rawDetail: string | null;
  } | null;
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
  const [validateLoading, setValidateLoading] = useState(false);
  const [validateResult, setValidateResult] = useState<{
    valid: boolean;
    errors: string[];
    rawDetail: string | null;
  } | null>(null);
  const {
    runDialogOpen,
    runDialogKind,
    previewRows,
    fullRows,
    runErrors,
    runSettings,
    previewLoading,
    fullLoading,
    executions,
    selectedExecutionId,
    setRunDialogOpen,
    setRunDialogKind,
    setPreviewRows,
    setFullRows,
    setRunErrors,
    setRunSettings,
    setPreviewLoading,
    setFullLoading,
    setExecutions,
    upsertExecution,
    selectExecution,
    resetForRecipe,
  } = useRecipeExecutionsStore(
    useShallow((state) => ({
      runDialogOpen: state.runDialogOpen,
      runDialogKind: state.runDialogKind,
      previewRows: state.previewRows,
      fullRows: state.fullRows,
      runErrors: state.runErrors,
      runSettings: state.runSettings,
      previewLoading: state.previewLoading,
      fullLoading: state.fullLoading,
      executions: state.executions,
      selectedExecutionId: state.selectedExecutionId,
      setRunDialogOpen: state.setRunDialogOpen,
      setRunDialogKind: state.setRunDialogKind,
      setPreviewRows: state.setPreviewRows,
      setFullRows: state.setFullRows,
      setRunErrors: state.setRunErrors,
      setRunSettings: state.setRunSettings,
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
          onSetPreviewErrors: setRunErrors,
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
    setRunErrors,
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

    setRunErrors(payloadResult.errors);
    toastError("Invalid recipe payload", payloadErrorMessage);
    return null;
  }, [payloadErrorMessage, payloadResult.errors, readPayload, setRunErrors]);

  const runExecution = useCallback(
    async (input: {
      kind: RecipeExecutionKind;
      payload: RecipePayload;
      rows: number;
      settings: RecipeRunSettings;
    }): Promise<boolean> => {
      const { kind, payload, rows, settings } = input;
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
      setRunDialogOpen(false);

      try {
        const jobPayload = buildExecutionPayload({
          payload,
          kind,
          rows,
          settings,
        });
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
          onSetPreviewErrors: setRunErrors,
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
        setRunErrors([message]);
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
      setPreviewLoading,
      setRunDialogOpen,
      setRunErrors,
      upsertAndPersist,
    ],
  );

  const runWithValidation = useCallback(
    async (kind: RecipeExecutionKind, rows: number): Promise<boolean> => {
      const payload = readExecutablePayload();
      if (!payload) {
        return false;
      }

      const normalizedRows = sanitizeExecutionRows(rows, kind);
      const executionPayload = buildExecutionPayload({
        payload,
        kind,
        rows: normalizedRows,
        settings: runSettings,
      });

      try {
        const validation = await validateRecipe(executionPayload);
        if (!validation.valid) {
          const errors = validation.errors.map((item) => item.message);
          const fallback = validation.raw_detail ?? "Validation failed.";
          const nextErrors = errors.length > 0 ? errors : [fallback];
          setRunErrors(nextErrors);
          toastError("Validation failed", nextErrors[0]);
          return false;
        }
      } catch (error) {
        const message = toErrorMessage(error, "Validation failed.");
        setRunErrors([message]);
        toastError("Validation failed", message);
        return false;
      }

      return runExecution({
        kind,
        payload,
        rows: normalizedRows,
        settings: runSettings,
      });
    },
    [readExecutablePayload, runExecution, runSettings, setRunErrors],
  );

  const runPreview = useCallback(async (): Promise<boolean> => {
    return runWithValidation("preview", previewRows);
  }, [previewRows, runWithValidation]);

  const runFull = useCallback(async (): Promise<boolean> => {
    return runWithValidation("full", fullRows);
  }, [fullRows, runWithValidation]);

  const runFromDialog = useCallback(async (): Promise<boolean> => {
    if (runDialogKind === "preview") {
      return runPreview();
    }
    return runFull();
  }, [runDialogKind, runFull, runPreview]);

  const validateFromDialog = useCallback(async (): Promise<boolean> => {
    const payload = readPayload();
    if (!payload) {
      const nextErrors = payloadResult.errors.length > 0
        ? payloadResult.errors
        : [payloadErrorMessage];
      setValidateResult({
        valid: false,
        errors: nextErrors,
        rawDetail: null,
      });
      return false;
    }

    const rows = runDialogKind === "preview" ? previewRows : fullRows;
    const normalizedRows = sanitizeExecutionRows(rows, runDialogKind);
    const executionPayload = buildExecutionPayload({
      payload,
      kind: runDialogKind,
      rows: normalizedRows,
      settings: runSettings,
    });

    setValidateLoading(true);
    try {
      const validation = await validateRecipe(executionPayload);
      const errors = validation.errors.map((item) => item.message);
      setValidateResult({
        valid: validation.valid,
        errors,
        rawDetail: validation.raw_detail ?? null,
      });
      return validation.valid;
    } catch (error) {
      const message = toErrorMessage(error, "Validation failed.");
      setValidateResult({
        valid: false,
        errors: [message],
        rawDetail: null,
      });
      return false;
    } finally {
      setValidateLoading(false);
    }
  }, [
    fullRows,
    payloadErrorMessage,
    payloadResult.errors,
    previewRows,
    readPayload,
    runDialogKind,
    runSettings,
  ]);

  const openRunDialog = useCallback(
    (kind: RecipeExecutionKind): void => {
      setRunErrors([]);
      setValidateResult(null);
      setRunDialogKind(kind);
      if (kind === "full") {
        const payload = readPayload();
        const payloadRows = Number(payload?.run?.rows);
        if (Number.isFinite(payloadRows) && payloadRows > 0) {
          setFullRows(Math.floor(payloadRows));
        }
      }
      setRunDialogOpen(true);
    },
    [
      readPayload,
      setFullRows,
      setRunDialogKind,
      setRunDialogOpen,
      setRunErrors,
    ],
  );

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
    runDialogOpen,
    runDialogKind,
    setRunDialogKind,
    setRunDialogOpen,
    previewRows,
    fullRows,
    setPreviewRows,
    setFullRows,
    runErrors,
    runSettings,
    setRunSettings,
    previewLoading,
    fullLoading,
    executions,
    selectedExecutionId,
    setSelectedExecutionId,
    openRunDialog,
    runFromDialog,
    validateFromDialog,
    validateLoading,
    validateResult,
    runPreview,
    runFull,
    cancelExecution,
    loadExecutionDatasetPage,
  };
}
