// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { toast } from "sonner";
import { toastError } from "@/shared/toast";
import {
  getInferenceStatus,
  loadModel,
} from "@/features/chat/api/chat-api";
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
  normalizeRunName,
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

/**
 * Auto-load the local model before running a recipe that uses it.
 *
 * Looks at payload.recipe.model_providers for any provider with is_local=true,
 * finds the bound model_configs and asks the backend to load whichever model
 * the first local-bound model_config points at. Skips when the inference
 * server already has that exact model active. This removes the "open /chat
 * first" prerequisite that users kept tripping on.
 */
async function ensureLocalModelLoaded(
  payload: RecipePayload,
): Promise<string | null> {
  const providers = Array.isArray(payload.recipe.model_providers)
    ? (payload.recipe.model_providers as Array<Record<string, unknown>>)
    : [];
  const localProviderNames = new Set<string>();
  for (const p of providers) {
    if (p.is_local === true && typeof p.name === "string") {
      localProviderNames.add(p.name);
    }
  }
  if (localProviderNames.size === 0) {
    return null;
  }
  const modelConfigs = Array.isArray(payload.recipe.model_configs)
    ? (payload.recipe.model_configs as Array<Record<string, unknown>>)
    : [];
  const boundConfig = modelConfigs.find(
    (c) => typeof c.provider === "string" && localProviderNames.has(c.provider),
  );
  const target =
    typeof boundConfig?.model === "string" ? boundConfig.model.trim() : "";
  if (!target) {
    return null;
  }

  try {
    const status = await getInferenceStatus();
    if (
      status.active_model &&
      status.active_model.toLowerCase() === target.toLowerCase()
    ) {
      return null;
    }
  } catch {
    // Fall through to load attempt; the backend will re-error if needed.
  }

  const toastId = toast.loading(`Loading ${target}…`, {
    description: "Starting the local inference server for this recipe.",
  });
  try {
    const isGguf = /gguf/i.test(target);
    await loadModel({
      model_path: target,
      hf_token: null,
      max_seq_length: isGguf ? 0 : 4096,
      load_in_4bit: true,
      is_lora: false,
      gguf_variant: null,
      trust_remote_code: false,
      chat_template_override: null,
      cache_type_kv: null,
      speculative_type: null,
    });
    toast.success(`Loaded ${target}`, { id: toastId, duration: 2000 });
    return null;
  } catch (error) {
    toast.dismiss(toastId);
    return error instanceof Error ? error.message : String(error);
  }
}

type UseRecipeExecutionsParams = {
  recipeId: string;
  currentSignature: string;
  payloadResult: RecipePayloadResult;
  initialRunRows?: number | null;
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
  fullRunName: string;
  setPreviewRows: (rows: number) => void;
  setFullRows: (rows: number) => void;
  setFullRunName: (name: string) => void;
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

function formatValidationMessages(input: {
  errors: Array<{ message: string; path?: string | null; code?: string | null }>;
}): string[] {
  return input.errors.map((item) => {
    const path = item.path?.trim();
    const code = item.code?.trim();
    const prefix = [
      code ? code.toUpperCase() : null,
      path ? `column ${path}` : null,
    ]
      .filter(Boolean)
      .join(" · ");
    return prefix ? `${prefix}: ${item.message}` : item.message;
  });
}

export function useRecipeExecutions({
  recipeId,
  currentSignature,
  payloadResult,
  initialRunRows,
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
    fullRunName,
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
    setFullRunName,
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
      fullRunName: state.fullRunName,
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
      setFullRunName: state.setFullRunName,
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

    // Seed previewRows from the recipe's original run.rows (read from the
    // loaded JSON, not the rebuilt payload (the builder hardcodes 5).
    // Templates ship their own suggested preview size (e.g. GitHub Support
    // Bot: 10); we honor it so users don't see a surprise 5.
    if (
      typeof initialRunRows === "number" &&
      Number.isFinite(initialRunRows) &&
      initialRunRows > 0 &&
      initialRunRows !== 5
    ) {
      setPreviewRows(Math.floor(initialRunRows));
    }

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
    initialRunRows,
    onPreviewSuccess,
    recipeId,
    resetForRecipe,
    setExecutions,
    setPreviewRows,
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
      runName: string | null;
    }): Promise<boolean> => {
      const { kind, payload, rows, settings, runName } = input;
      const setLoading = kind === "preview" ? setPreviewLoading : setFullLoading;
      const label = executionLabel(kind);

      setLoading(true);
      const baseExecution = createBaseExecutionRecord({
        recipeId,
        kind,
        rows,
        currentSignature,
        runName,
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
          runName,
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
    async (
      kind: RecipeExecutionKind,
      rows: number,
      runName: string | null,
    ): Promise<boolean> => {
      const trimmedRunName = typeof runName === "string" ? runName.trim() : "";
      if (kind === "full" && !trimmedRunName) {
        const message = "Run name required for full runs.";
        setRunErrors([message]);
        toastError("Run name required", message);
        return false;
      }

      const payload = readExecutablePayload();
      if (!payload) {
        return false;
      }

      // Flip to the Runs pane BEFORE we run ensureLocalModelLoaded + validate.
      // Validation re-crawls the seed (multiple seconds for the github_repo
      // reader) and the user otherwise stares at a "Running..." button with
      // nothing else changing. runExecution() later no-ops this callback if
      // the view has already been flipped, so we fire it once here.
      onExecutionStart?.();

      const localLoadError = await ensureLocalModelLoaded(payload);
      if (localLoadError) {
        setRunErrors([localLoadError]);
        toastError("Local model failed to load", localLoadError);
        return false;
      }

      const normalizedRows = sanitizeExecutionRows(rows, kind);
      const executionPayload = buildExecutionPayload({
        payload,
        kind,
        rows: normalizedRows,
        settings: runSettings,
        runName,
      });

      try {
        const validation = await validateRecipe(executionPayload);
        if (!validation.valid) {
          const errors = formatValidationMessages({ errors: validation.errors });
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
        runName,
      });
    },
    [
      onExecutionStart,
      readExecutablePayload,
      runExecution,
      runSettings,
      setRunErrors,
    ],
  );

  const runPreview = useCallback(async (): Promise<boolean> => {
    return runWithValidation("preview", previewRows, null);
  }, [previewRows, runWithValidation]);

  const runFull = useCallback(async (): Promise<boolean> => {
    return runWithValidation("full", fullRows, fullRunName);
  }, [fullRows, fullRunName, runWithValidation]);

  const runFromDialog = useCallback(async (): Promise<boolean> => {
    setValidateResult(null);
    if (runDialogKind === "preview") {
      return runPreview();
    }
    return runFull();
  }, [runDialogKind, runFull, runPreview]);

  const validateFromDialog = useCallback(async (): Promise<boolean> => {
    setRunErrors([]);
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
      runName: runDialogKind === "full" ? normalizeRunName(fullRunName) : null,
    });

    setValidateLoading(true);
    try {
      const validation = await validateRecipe(executionPayload);
      const errors = formatValidationMessages({ errors: validation.errors });
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
    fullRunName,
    fullRows,
    payloadErrorMessage,
    payloadResult.errors,
    previewRows,
    readPayload,
    runDialogKind,
    runSettings,
    setRunErrors,
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
    fullRunName,
    setPreviewRows,
    setFullRows,
    setFullRunName,
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
