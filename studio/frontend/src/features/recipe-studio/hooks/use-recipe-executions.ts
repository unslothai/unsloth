// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthSubjectKey, subscribeAuthSubject } from "@/features/auth";
import { getInferenceStatus, loadModel } from "@/features/chat";
import { toast } from "@/lib/toast";
import { toastError } from "@/shared/toast";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { useShallow } from "zustand/react/shallow";
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
  normalizeRunName,
  toErrorMessage,
  withExecutionDefaults,
} from "../executions/execution-helpers";
import {
  findResumableExecution,
  hydrateCompletedFullExecutionDataset,
  loadSortedRecipeExecutionPage,
} from "../executions/hydration";
import {
  buildExecutionPayload,
  sanitizeExecutionRows,
} from "../executions/run-settings";
import { createBaseExecutionRecord } from "../executions/runtime";
import { trackRecipeExecution } from "../executions/tracker";
import {
  type RecipeRunSettings,
  useRecipeExecutionsStore,
} from "../stores/recipe-executions";
import type {
  RecipePayload,
  RecipePayloadResult,
} from "../utils/payload/types";

const GGUF_MODEL_PATTERN = /gguf/i;

type ExecutionHistoryOwner = {
  subjectKey: string;
  recipeId: string;
  generation: number;
  lifecycle: object;
};

type ExecutionHistoryState = {
  owner: ExecutionHistoryOwner | null;
  cursor: string | null;
  loadingOlder: boolean;
};

function sameExecutionHistoryOwner(
  left: ExecutionHistoryOwner | null,
  right: ExecutionHistoryOwner,
): boolean {
  return Boolean(
    left === right &&
      left &&
      left.subjectKey === right.subjectKey &&
      left.recipeId === right.recipeId &&
      left.generation === right.generation,
  );
}

function getServerAuthSubjectKey(): string {
  return "anonymous";
}

function collectUsedLlmModelAliases(payload: RecipePayload): Set<string> {
  const columns = Array.isArray(payload.recipe.columns)
    ? payload.recipe.columns
    : [];
  const aliases = new Set<string>();
  for (const column of columns) {
    const columnType = column.column_type;
    if (typeof columnType !== "string" || !columnType.startsWith("llm-")) {
      continue;
    }
    const alias = column.model_alias;
    if (typeof alias === "string" && alias.trim()) {
      aliases.add(alias.trim());
    }
  }
  return aliases;
}

type LocalModelSelection = {
  target: string;
  ggufVariant: string;
  aliases: string[];
};

type LocalModelLoadPlan =
  | { selection: LocalModelSelection; error: null; legacyAliases?: never }
  | { selection: null; error: string; legacyAliases?: never }
  | { selection: null; error: null; legacyAliases: string[] };

type RestorableLocalModelSnapshot = {
  selection: LocalModelSelection | null;
  unrestorableLabel: string | null;
};

function getLocalProviderNames(payload: RecipePayload): Set<string> {
  const providers = Array.isArray(payload.recipe.model_providers)
    ? (payload.recipe.model_providers as Record<string, unknown>[])
    : [];
  const localProviderNames = new Set<string>();
  for (const provider of providers) {
    if (provider.is_local === true && typeof provider.name === "string") {
      localProviderNames.add(provider.name);
    }
  }
  return localProviderNames;
}

function findUsedLocalModelConfigs(
  payload: RecipePayload,
  localProviderNames: Set<string>,
): Record<string, unknown>[] {
  const usedAliases = collectUsedLlmModelAliases(payload);
  if (usedAliases.size === 0) {
    return [];
  }
  const modelConfigs = Array.isArray(payload.recipe.model_configs)
    ? payload.recipe.model_configs
    : [];
  return modelConfigs.filter((config) => {
    const provider = config.provider;
    const alias = config.alias;
    return (
      typeof provider === "string" &&
      localProviderNames.has(provider) &&
      typeof alias === "string" &&
      usedAliases.has(alias)
    );
  });
}

function readLocalModelSelection(
  boundConfig: Record<string, unknown>,
): LocalModelLoadPlan {
  const alias =
    typeof boundConfig.alias === "string" ? boundConfig.alias : "local model";
  const target =
    typeof boundConfig.model === "string" ? boundConfig.model.trim() : "";
  const ggufVariant =
    typeof boundConfig.gguf_variant === "string"
      ? boundConfig.gguf_variant.trim()
      : "";
  if (!target) {
    return {
      selection: null,
      error: `Model config ${alias}: choose a local model before validating or running this recipe.`,
    };
  }
  if (target.toLowerCase() === "local") {
    return { selection: null, error: null, legacyAliases: [alias] };
  }
  return { selection: { target, ggufVariant, aliases: [alias] }, error: null };
}

function getLocalModelLoadPlan(
  boundConfigs: Record<string, unknown>[],
): LocalModelLoadPlan | null {
  const selections = new Map<string, LocalModelSelection>();
  const legacyAliases: string[] = [];
  for (const boundConfig of boundConfigs) {
    const next = readLocalModelSelection(boundConfig);
    if (next.error) {
      return next;
    }
    if (next.legacyAliases) {
      legacyAliases.push(...next.legacyAliases);
      continue;
    }
    const selection = next.selection;
    if (!selection) {
      continue;
    }
    const key = `${selection.target.toLowerCase()}\u0000${selection.ggufVariant}`;
    const existing = selections.get(key);
    if (existing) {
      existing.aliases.push(...selection.aliases);
      continue;
    }
    selections.set(key, selection);
  }

  if (legacyAliases.length > 0 && selections.size > 0) {
    const aliases = [
      ...legacyAliases,
      ...[...selections.values()].flatMap((selection) => selection.aliases),
    ].join(", ");
    return {
      selection: null,
      error: `Recipes found mixed legacy and selected local models. Reselect the same concrete local model for: ${aliases}.`,
    };
  }

  if (legacyAliases.length > 0) {
    return { selection: null, error: null, legacyAliases };
  }

  if (selections.size > 1) {
    const aliases = [...selections.values()]
      .flatMap((selection) => selection.aliases)
      .join(", ");
    return {
      selection: null,
      error: `Recipes supports one active local model per run. Select the same local model and GGUF variant for: ${aliases}.`,
    };
  }

  const selection = [...selections.values()][0];
  return selection ? { selection, error: null } : null;
}

function isDirectGgufTarget(target: string): boolean {
  return target.toLowerCase().endsWith(".gguf");
}

function localSelectionMatchesActive(input: {
  target: string;
  ggufVariant: string;
  activeModel: string | null | undefined;
  activeVariant: string;
}): boolean {
  const { target, ggufVariant, activeModel, activeVariant } = input;
  if (!activeModel || activeModel.toLowerCase() !== target.toLowerCase()) {
    return false;
  }
  return (
    activeVariant === ggufVariant ||
    (isDirectGgufTarget(target) && !ggufVariant)
  );
}

async function isLocalModelAlreadyLoaded(
  selection: LocalModelSelection,
): Promise<boolean> {
  const { target, ggufVariant } = selection;
  try {
    const status = await getInferenceStatus();
    return localSelectionMatchesActive({
      target,
      ggufVariant,
      activeModel: status.model_identifier ?? status.active_model,
      activeVariant: status.gguf_variant?.trim() ?? "",
    });
  } catch {
    // Fall through to load attempt; the backend will re-error if needed.
    return false;
  }
}

async function loadLocalModelSelection(
  selection: LocalModelSelection,
): Promise<string | null> {
  const { target, ggufVariant } = selection;
  const modelLabel = ggufVariant ? `${target} (${ggufVariant})` : target;
  const toastId = toast.loading(`Loading ${modelLabel}...`, {
    description: "Starting the local inference server for this recipe.",
  });
  try {
    const isGguf = GGUF_MODEL_PATTERN.test(target) || Boolean(ggufVariant);
    await loadModel({
      // biome-ignore lint/style/useNamingConvention: api schema
      model_path: target,
      // biome-ignore lint/style/useNamingConvention: api schema
      hf_token: null,
      // biome-ignore lint/style/useNamingConvention: api schema
      max_seq_length: isGguf ? 0 : 4096,
      // biome-ignore lint/style/useNamingConvention: api schema
      load_in_4bit: true,
      // biome-ignore lint/style/useNamingConvention: api schema
      is_lora: false,
      // biome-ignore lint/style/useNamingConvention: api schema
      gguf_variant: ggufVariant || null,
      // biome-ignore lint/style/useNamingConvention: api schema
      trust_remote_code: false,
      // biome-ignore lint/style/useNamingConvention: api schema
      chat_template_override: null,
      // biome-ignore lint/style/useNamingConvention: api schema
      cache_type_kv: null,
      // biome-ignore lint/style/useNamingConvention: api schema
      speculative_type: null,
      // biome-ignore lint/style/useNamingConvention: api schema
      tensor_parallel: false,
    });
    toast.success(`Loaded ${modelLabel}`, { id: toastId, duration: 2000 });
    return null;
  } catch (error) {
    toast.dismiss(toastId);
    return error instanceof Error ? error.message : String(error);
  }
}

function getLocalModelLoadPlanForPayload(
  payload: RecipePayload,
): LocalModelLoadPlan | null {
  const localProviderNames = getLocalProviderNames(payload);
  if (localProviderNames.size === 0) {
    return null;
  }

  const boundConfigs = findUsedLocalModelConfigs(payload, localProviderNames);
  return getLocalModelLoadPlan(boundConfigs);
}

async function getActiveLocalModelSelection(): Promise<LocalModelSelection | null> {
  try {
    const status = await getInferenceStatus();
    const target = status.active_model?.trim();
    if (!target) {
      return null;
    }
    return {
      target,
      ggufVariant: status.gguf_variant?.trim() ?? "",
      aliases: ["previous Chat model"],
    };
  } catch {
    return null;
  }
}

async function getRestorableActiveLocalModelSelection(): Promise<RestorableLocalModelSnapshot> {
  try {
    const status = await getInferenceStatus();
    const activeLabel = status.active_model?.trim() ?? null;
    const target = (
      status.model_identifier ?? (status.is_gguf ? null : status.active_model)
    )?.trim();
    if (!target) {
      return {
        selection: null,
        unrestorableLabel: activeLabel,
      };
    }
    return {
      selection: {
        target,
        ggufVariant: status.gguf_variant?.trim() ?? "",
        aliases: ["previous Chat model"],
      },
      unrestorableLabel: null,
    };
  } catch {
    return { selection: null, unrestorableLabel: null };
  }
}

function isSameLocalModelSelection(
  left: LocalModelSelection | null,
  right: LocalModelSelection,
): boolean {
  return Boolean(
    left &&
      left.target.toLowerCase() === right.target.toLowerCase() &&
      left.ggufVariant === right.ggufVariant,
  );
}

async function ensureLocalModelLoaded(
  payload: RecipePayload,
): Promise<string | null> {
  const loadPlan = getLocalModelLoadPlanForPayload(payload);
  if (!loadPlan) {
    return null;
  }
  if (loadPlan.legacyAliases) {
    const activeSelection = await getActiveLocalModelSelection();
    return activeSelection
      ? null
      : `Existing recipe uses legacy local model for ${loadPlan.legacyAliases.join(", ")}. Select a concrete local model or load one in Chat.`;
  }
  if (!loadPlan.selection) {
    return loadPlan.error;
  }
  if (await isLocalModelAlreadyLoaded(loadPlan.selection)) {
    return null;
  }
  return loadLocalModelSelection(loadPlan.selection);
}

async function prepareLocalModelForRun(payload: RecipePayload): Promise<{
  error: string | null;
  restorePrevious: (() => Promise<void>) | null;
}> {
  const loadPlan = getLocalModelLoadPlanForPayload(payload);
  if (!loadPlan) {
    return { error: null, restorePrevious: null };
  }
  if (loadPlan.legacyAliases) {
    const activeSelection = await getActiveLocalModelSelection();
    return activeSelection
      ? { error: null, restorePrevious: null }
      : {
          error: `Existing recipe uses legacy local model for ${loadPlan.legacyAliases.join(", ")}. Select a concrete local model or load one in Chat.`,
          restorePrevious: null,
        };
  }
  if (!loadPlan.selection) {
    return { error: loadPlan.error, restorePrevious: null };
  }
  if (await isLocalModelAlreadyLoaded(loadPlan.selection)) {
    return { error: null, restorePrevious: null };
  }

  const previousSnapshot = await getRestorableActiveLocalModelSelection();
  const previousSelection = previousSnapshot.selection;
  const error = await loadLocalModelSelection(loadPlan.selection);
  if (error) {
    return { error, restorePrevious: null };
  }
  if (isSameLocalModelSelection(previousSelection, loadPlan.selection)) {
    return { error: null, restorePrevious: null };
  }
  return {
    error: null,
    restorePrevious: previousSelection
      ? async () => {
          const restoreError = await loadLocalModelSelection(previousSelection);
          if (restoreError) {
            toastError("Could not restore previous local model", restoreError);
          }
        }
      : previousSnapshot.unrestorableLabel
        ? () => {
            toast.warning("Previous local model was not restored", {
              description: `${previousSnapshot.unrestorableLabel} was selected from a native file path. Reopen it in Chat to continue with that model.`,
            });
            return Promise.resolve();
          }
        : null,
  };
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
  hasOlderExecutions: boolean;
  olderExecutionsLoading: boolean;
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
  loadOlderExecutions: () => Promise<void>;
  loadExecutionDatasetPage: (id: string, page: number) => Promise<void>;
};

function formatValidationMessages(input: {
  errors: Array<{
    message: string;
    path?: string | null;
    code?: string | null;
  }>;
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
  const authSubjectKey = useSyncExternalStore(
    subscribeAuthSubject,
    getAuthSubjectKey,
    getServerAuthSubjectKey,
  );
  const [validateLoading, setValidateLoading] = useState(false);
  const [executionHistory, setExecutionHistory] =
    useState<ExecutionHistoryState>({
      owner: null,
      cursor: null,
      loadingOlder: false,
    });
  const historyGenerationRef = useRef(0);
  const olderHistoryRequestRef = useRef<object | null>(null);
  const datasetHydrationRequestsRef = useRef(new Set<string>());
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
    mergeExecutions,
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
      mergeExecutions: state.mergeExecutions,
      upsertExecution: state.upsertExecution,
      selectExecution: state.selectExecution,
      resetForRecipe: state.resetForRecipe,
    })),
  );
  const payloadErrorMessage = payloadResult.errors[0] ?? "Invalid payload.";

  const historyLifecycle = useMemo(
    () => ({
      subjectKey: authSubjectKey,
      recipeId,
      initialRunRows,
      onPreviewSuccess,
      resetForRecipe,
      setExecutions,
      setPreviewRows,
      setRunErrors,
      upsertExecution,
    }),
    [
      authSubjectKey,
      initialRunRows,
      onPreviewSuccess,
      recipeId,
      resetForRecipe,
      setExecutions,
      setPreviewRows,
      setRunErrors,
      upsertExecution,
    ],
  );
  const historyLifecycleRef = useRef(historyLifecycle);
  useLayoutEffect(() => {
    historyLifecycleRef.current = historyLifecycle;
  }, [historyLifecycle]);
  const activeExecutionHistory =
    executionHistory.owner?.lifecycle === historyLifecycle
      ? executionHistory
      : null;
  const executionHistoryCursor = activeExecutionHistory?.cursor ?? null;
  const olderExecutionsLoading = activeExecutionHistory?.loadingOlder ?? false;
  const visibleExecutions = activeExecutionHistory ? executions : [];
  const visibleSelectedExecutionId = activeExecutionHistory
    ? selectedExecutionId
    : null;

  const isExecutionOwnerActive = useCallback(
    (owner: ExecutionHistoryOwner): boolean =>
      historyGenerationRef.current === owner.generation &&
      getAuthSubjectKey() === owner.subjectKey &&
      historyLifecycleRef.current === owner.lifecycle,
    [],
  );

  const upsertAndPersistForOwner = useCallback(
    (owner: ExecutionHistoryOwner, record: RecipeExecutionRecord): void => {
      if (
        !isExecutionOwnerActive(owner) ||
        record.recipeId !== owner.recipeId
      ) {
        return;
      }
      const normalizedRecord = withExecutionDefaults(record);
      upsertExecution(normalizedRecord);
      saveRecipeExecution(normalizedRecord, {
        subjectKey: owner.subjectKey,
        recipeId: owner.recipeId,
        generation: owner.generation,
        isCurrent: () => isExecutionOwnerActive(owner),
      }).catch((error) => {
        // biome-ignore lint/suspicious/noConsole: background persistence failures should not interrupt the UI
        console.error("Save recipe execution failed:", error);
      });
    },
    [isExecutionOwnerActive, upsertExecution],
  );

  useEffect(() => {
    let cancelled = false;
    const owner: ExecutionHistoryOwner = {
      subjectKey: historyLifecycle.subjectKey,
      recipeId: historyLifecycle.recipeId,
      generation: historyGenerationRef.current + 1,
      lifecycle: historyLifecycle,
    };
    historyGenerationRef.current = owner.generation;
    olderHistoryRequestRef.current = null;
    const datasetHydrationRequests = datasetHydrationRequestsRef.current;
    datasetHydrationRequests.clear();

    const isActive = (): boolean =>
      !cancelled &&
      historyGenerationRef.current === owner.generation &&
      getAuthSubjectKey() === owner.subjectKey &&
      historyLifecycleRef.current === owner.lifecycle;

    historyLifecycle.resetForRecipe();

    // Seed previewRows from the recipe's original run.rows (the loaded JSON, not
    // the rebuilt payload, which hardcodes 5). Templates ship a suggested preview
    // size (e.g. GitHub Support Bot: 10); honor it so users don't see a surprise 5.
    if (
      typeof historyLifecycle.initialRunRows === "number" &&
      Number.isFinite(historyLifecycle.initialRunRows) &&
      historyLifecycle.initialRunRows > 0 &&
      historyLifecycle.initialRunRows !== 5
    ) {
      historyLifecycle.setPreviewRows(
        Math.floor(historyLifecycle.initialRunRows),
      );
    }

    async function hydrate(): Promise<void> {
      try {
        const page = await loadSortedRecipeExecutionPage(owner.recipeId);
        if (!isActive()) {
          return;
        }

        historyLifecycle.setExecutions(page.executions);
        setExecutionHistory({
          owner,
          cursor: page.nextCursor,
          loadingOlder: false,
        });
        const initiallySelected = page.executions[0];
        if (
          initiallySelected?.kind === "full" &&
          initiallySelected.status === "completed" &&
          initiallySelected.jobId
        ) {
          const requestKey = `${owner.generation}:${initiallySelected.id}`;
          datasetHydrationRequests.add(requestKey);
          void hydrateCompletedFullExecutionDataset(initiallySelected)
            .then((hydrated) => {
              if (
                isActive() &&
                useRecipeExecutionsStore.getState().selectedExecutionId ===
                  initiallySelected.id
              ) {
                // Hydrating memory-only pages
                // must not persist metadata.
                historyLifecycle.upsertExecution(hydrated);
              }
            })
            .catch((error) => {
              if (isActive()) {
                // biome-ignore lint/suspicious/noConsole: hydration failure is a non-blocking diagnostic
                console.error("Load execution dataset page failed:", error);
              }
            })
            .finally(() => {
              datasetHydrationRequests.delete(requestKey);
            });
        }
        const resumable = findResumableExecution(page.executions);
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
          onUpsert: (record) => {
            if (isActive()) upsertAndPersistForOwner(owner, record);
          },
          onSetPreviewErrors: (errors) => {
            if (isActive()) historyLifecycle.setRunErrors(errors);
          },
          onPreviewSuccess: () => {
            if (isActive()) historyLifecycle.onPreviewSuccess?.();
          },
        });
      } catch (error) {
        if (!isActive()) return;
        // Keep the cleared owner store active so outages neither hide new runs nor expose stale data.
        historyLifecycle.setExecutions([]);
        setExecutionHistory({ owner, cursor: null, loadingOlder: false });
        // biome-ignore lint/suspicious/noConsole: hydration failures are non-blocking diagnostics
        console.error("Load recipe executions failed:", error);
      }
    }

    hydrate();

    return () => {
      cancelled = true;
      if (historyGenerationRef.current === owner.generation) {
        historyGenerationRef.current += 1;
      }
      olderHistoryRequestRef.current = null;
      datasetHydrationRequests.clear();
    };
  }, [historyLifecycle, upsertAndPersistForOwner]);

  const loadOlderExecutions = useCallback(async (): Promise<void> => {
    const owner = activeExecutionHistory?.owner;
    if (
      !owner ||
      !executionHistoryCursor ||
      olderExecutionsLoading ||
      olderHistoryRequestRef.current
    ) {
      return;
    }
    const cursor = executionHistoryCursor;
    const request = {};
    olderHistoryRequestRef.current = request;
    const isActive = (): boolean =>
      historyGenerationRef.current === owner.generation &&
      getAuthSubjectKey() === owner.subjectKey &&
      historyLifecycleRef.current === owner.lifecycle;
    setExecutionHistory((current) =>
      sameExecutionHistoryOwner(current.owner, owner)
        ? { ...current, loadingOlder: true }
        : current,
    );
    try {
      const page = await loadSortedRecipeExecutionPage(owner.recipeId, cursor);
      if (!isActive()) {
        return;
      }
      mergeExecutions(page.executions);
      setExecutionHistory((current) =>
        sameExecutionHistoryOwner(current.owner, owner)
          ? { ...current, cursor: page.nextCursor }
          : current,
      );
    } catch (error) {
      if (!isActive()) return;
      toastError(
        "Could not load older runs",
        toErrorMessage(error, "Execution history could not be loaded."),
      );
    } finally {
      if (olderHistoryRequestRef.current === request) {
        olderHistoryRequestRef.current = null;
        if (isActive()) {
          setExecutionHistory((current) =>
            sameExecutionHistoryOwner(current.owner, owner)
              ? { ...current, loadingOlder: false }
              : current,
          );
        }
      }
    }
  }, [
    activeExecutionHistory,
    executionHistoryCursor,
    mergeExecutions,
    olderExecutionsLoading,
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
      restorePrevious?: (() => Promise<void>) | null;
    }): Promise<boolean> => {
      const { kind, payload, rows, settings, runName, restorePrevious } = input;
      const owner = activeExecutionHistory?.owner;
      if (!owner || !isExecutionOwnerActive(owner)) {
        toastError(
          "Runs are still loading",
          "Wait for this recipe's run history to finish loading and try again.",
        );
        return false;
      }
      const ownedUpsert = (record: RecipeExecutionRecord): void => {
        upsertAndPersistForOwner(owner, record);
      };
      const setOwnedRunErrors = (errors: string[]): void => {
        if (isExecutionOwnerActive(owner)) setRunErrors(errors);
      };
      const setLoading =
        kind === "preview" ? setPreviewLoading : setFullLoading;
      const label = executionLabel(kind);

      setLoading(true);
      const baseExecution = createBaseExecutionRecord({
        recipeId,
        kind,
        rows,
        currentSignature,
        runName,
      });

      ownedUpsert(baseExecution);
      onExecutionStart?.();
      setRunDialogOpen(false);

      let jobCreated = false;
      let shouldRestorePrevious = false;
      try {
        const jobPayload = buildExecutionPayload({
          payload,
          kind,
          rows,
          settings,
          runName,
        });
        const createdJob = await createRecipeJob(jobPayload, {
          expectedSubjectKey: owner.subjectKey,
        });
        jobCreated = true;
        const executionWithJob = {
          ...baseExecution,
          jobId: createdJob.job_id,
        };
        ownedUpsert(executionWithJob);

        const tracked = await trackRecipeExecution({
          label,
          kind,
          rows,
          jobId: createdJob.job_id,
          initialExecution: executionWithJob,
          notify: true,
          onUpsert: ownedUpsert,
          onSetPreviewErrors: setOwnedRunErrors,
          onPreviewSuccess: () => {
            if (isExecutionOwnerActive(owner)) onPreviewSuccess?.();
          },
        });
        shouldRestorePrevious = tracked.terminal;
        return tracked.success;
      } catch (error) {
        const message = toErrorMessage(error, `${label} request failed.`);
        if (isExecutionOwnerActive(owner)) {
          ownedUpsert({
            ...baseExecution,
            status: "error",
            error: message,
            finishedAt: Date.now(),
          });
          setRunErrors([message]);
          toastError(`${label} failed`, message);
        }
        if (!jobCreated) {
          shouldRestorePrevious = true;
        }
        return false;
      } finally {
        if (shouldRestorePrevious && restorePrevious) {
          await restorePrevious();
        }
        if (isExecutionOwnerActive(owner)) setLoading(false);
      }
    },
    [
      activeExecutionHistory,
      currentSignature,
      isExecutionOwnerActive,
      onExecutionStart,
      onPreviewSuccess,
      recipeId,
      setFullLoading,
      setPreviewLoading,
      setRunDialogOpen,
      setRunErrors,
      upsertAndPersistForOwner,
    ],
  );

  const prepareLocalModelForExecution = useCallback(
    async (
      payload: RecipePayload,
    ): Promise<(() => Promise<void>) | null | false> => {
      const { error, restorePrevious } = await prepareLocalModelForRun(payload);
      if (!error) {
        return restorePrevious;
      }
      setRunErrors([error]);
      toastError("Local model failed to load", error);
      return false;
    },
    [setRunErrors],
  );

  const validateExecutionPayload = useCallback(
    async (
      executionPayload: Parameters<typeof validateRecipe>[0],
    ): Promise<boolean> => {
      try {
        const validation = await validateRecipe(executionPayload);
        if (validation.valid) {
          return true;
        }
        const errors = formatValidationMessages({
          errors: validation.errors,
        });
        const fallback = validation.raw_detail ?? "Validation failed.";
        const nextErrors = errors.length > 0 ? errors : [fallback];
        setRunErrors(nextErrors);
        toastError("Validation failed", nextErrors[0]);
        return false;
      } catch (error) {
        const message = toErrorMessage(error, "Validation failed.");
        setRunErrors([message]);
        toastError("Validation failed", message);
        return false;
      }
    },
    [setRunErrors],
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

      // Flip to the Runs pane before validation starts. Validation can re-crawl
      // the seed (seconds for the github_repo reader); runExecution() later no-ops
      // this callback if the view has already been flipped.
      onExecutionStart?.();

      const normalizedRows = sanitizeExecutionRows(rows, kind);
      const executionPayload = buildExecutionPayload({
        payload,
        kind,
        rows: normalizedRows,
        settings: runSettings,
        runName,
      });

      if (!(await validateExecutionPayload(executionPayload))) {
        return false;
      }

      // Recipe and Chat share one singleton local inference backend. This direct
      // load is a point-in-time handoff to job creation, not a lease: if Chat
      // swaps models after this succeeds, the backend runs against current state.
      // A future generation token should be validated across this load and the
      // `/jobs` loaded-model gate.
      const restorePrevious = await prepareLocalModelForExecution(payload);
      if (restorePrevious === false) {
        return false;
      }

      return runExecution({
        kind,
        payload,
        rows: normalizedRows,
        settings: runSettings,
        runName,
        restorePrevious,
      });
    },
    [
      onExecutionStart,
      prepareLocalModelForExecution,
      readExecutablePayload,
      runExecution,
      runSettings,
      setRunErrors,
      validateExecutionPayload,
    ],
  );

  const runPreview = useCallback((): Promise<boolean> => {
    return runWithValidation("preview", previewRows, null);
  }, [previewRows, runWithValidation]);

  const runFull = useCallback((): Promise<boolean> => {
    return runWithValidation("full", fullRows, fullRunName);
  }, [fullRows, fullRunName, runWithValidation]);

  const runFromDialog = useCallback((): Promise<boolean> => {
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
      const nextErrors =
        payloadResult.errors.length > 0
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

    setValidateLoading(true);
    try {
      const executionPayload = buildExecutionPayload({
        payload,
        kind: runDialogKind,
        rows: normalizedRows,
        settings: runSettings,
        runName:
          runDialogKind === "full" ? normalizeRunName(fullRunName) : null,
      });
      const validation = await validateRecipe(executionPayload);
      const errors = formatValidationMessages({ errors: validation.errors });
      if (!validation.valid) {
        setValidateResult({
          valid: false,
          errors,
          rawDetail: validation.raw_detail ?? null,
        });
        return false;
      }

      const localLoadError = await ensureLocalModelLoaded(payload);
      if (localLoadError) {
        setRunErrors([localLoadError]);
        setValidateResult({
          valid: false,
          errors: [localLoadError],
          rawDetail: null,
        });
        toastError("Local model failed to load", localLoadError);
        return false;
      }

      setValidateResult({
        valid: true,
        errors,
        rawDetail: validation.raw_detail ?? null,
      });
      return true;
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
      const owner = activeExecutionHistory?.owner;
      const execution = owner
        ? executions.find((entry) => entry.id === id)
        : undefined;
      if (
        !owner ||
        !isExecutionOwnerActive(owner) ||
        execution?.recipeId !== owner.recipeId ||
        !execution.jobId
      ) {
        return;
      }
      try {
        await cancelRecipeJob(execution.jobId);
        upsertAndPersistForOwner(owner, {
          ...execution,
          status: "cancelling",
        });
      } catch (error) {
        if (isExecutionOwnerActive(owner)) {
          const message = toErrorMessage(error, "Could not cancel execution.");
          toastError("Cancel failed", message);
        }
      }
    },
    [
      activeExecutionHistory,
      executions,
      isExecutionOwnerActive,
      upsertAndPersistForOwner,
    ],
  );

  const loadExecutionDatasetPage = useCallback(
    async (id: string, page: number): Promise<void> => {
      const owner = activeExecutionHistory?.owner;
      const execution = owner
        ? executions.find((entry) => entry.id === id)
        : undefined;
      if (
        !owner ||
        !isExecutionOwnerActive(owner) ||
        !execution ||
        execution.recipeId !== owner.recipeId ||
        execution.kind !== "full" ||
        !execution.jobId ||
        page < 1
      ) {
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
          typeof response.total === "number"
            ? response.total
            : execution.datasetTotal;
        upsertAndPersistForOwner(owner, {
          ...execution,
          dataset,
          datasetTotal: total,
          datasetPage: page,
        });
      } catch (error) {
        if (isExecutionOwnerActive(owner)) {
          const message = toErrorMessage(error, "Could not load dataset page.");
          toastError("Dataset page failed", message);
        }
      }
    },
    [
      activeExecutionHistory,
      executions,
      isExecutionOwnerActive,
      upsertAndPersistForOwner,
    ],
  );

  const setSelectedExecutionId = useCallback(
    (id: string): void => {
      selectExecution(id);
      const owner = activeExecutionHistory?.owner;
      const execution = owner
        ? executions.find((entry) => entry.id === id)
        : undefined;
      const requestKey = owner ? `${owner.generation}:${id}` : id;
      if (
        !owner ||
        !execution ||
        execution.kind !== "full" ||
        execution.status !== "completed" ||
        !execution.jobId ||
        execution.dataset.length > 0 ||
        datasetHydrationRequestsRef.current.has(requestKey)
      ) {
        return;
      }

      datasetHydrationRequestsRef.current.add(requestKey);
      void hydrateCompletedFullExecutionDataset(execution)
        .then((hydrated) => {
          const stillActive =
            historyGenerationRef.current === owner.generation &&
            getAuthSubjectKey() === owner.subjectKey &&
            useRecipeExecutionsStore.getState().selectedExecutionId === id;
          if (stillActive) {
            // Hydrating memory-only pages
            // must not persist metadata.
            upsertExecution(hydrated);
          }
        })
        .catch((error) => {
          if (
            historyGenerationRef.current === owner.generation &&
            getAuthSubjectKey() === owner.subjectKey
          ) {
            toastError(
              "Dataset page failed",
              toErrorMessage(error, "Could not load dataset page."),
            );
          }
        })
        .finally(() => {
          datasetHydrationRequestsRef.current.delete(requestKey);
        });
    },
    [activeExecutionHistory, executions, selectExecution, upsertExecution],
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
    executions: visibleExecutions,
    hasOlderExecutions: executionHistoryCursor !== null,
    olderExecutionsLoading,
    selectedExecutionId: visibleSelectedExecutionId,
    setSelectedExecutionId,
    openRunDialog,
    runFromDialog,
    validateFromDialog,
    validateLoading,
    validateResult,
    runPreview,
    runFull,
    cancelExecution,
    loadOlderExecutions,
    loadExecutionDatasetPage,
  };
}
