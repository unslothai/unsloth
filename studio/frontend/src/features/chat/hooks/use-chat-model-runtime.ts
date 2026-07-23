// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createElement, useCallback, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import {
  confirmRemoteCodeIfNeeded,
  useRemoteCodeConsentDialogStore,
} from "@/features/security";
import { useHfTokenWarningStore } from "@/features/hf-auth";
import {
  confirmTransformersUpgradeIfNeeded,
  useTransformersUpgradeDialogStore,
} from "@/features/transformers-upgrade";
import { consumeNativePathToken } from "@/features/native-intents/api";
import {
  notifyNative,
  primeNativeNotificationPermission,
  safeNotificationLabel,
  sanitizeNotificationBody,
} from "@/lib/native-notifications";
import { ModelLoadDescription } from "../components/model-load-status";
import {
  getDownloadProgress,
  getGgufDownloadProgress,
  getInferenceStatus,
  getLoadProgress,
  listLoras,
  listModels,
  loadModel,
  unloadModel,
  validateModel,
} from "../api/chat-api";
import { formatEta, formatRate } from "../utils/format-transfer";
import {
  GPU_LAYERS_AUTO,
  isLocalModelPath,
  loadedGpuMemoryFields,
  persistGpuMemoryModeOnLoad,
  readPersistedSpeculativeType,
  reconcilePersistedGpuIds,
  resolveToolsEnabledOnLoad,
  saveSpeculativeType,
  useChatRuntimeStore,
  type LoadingModelPick,
  type ReasoningEffort,
} from "../stores/chat-runtime-store";
import { clampReasoningEffortToLevels } from "../provider-capabilities";
import {
  applyActiveModelStatusToStore,
  clampLocalReasoningEffort,
  normalizeSpeculativeType,
  resolveInferenceCheckpointId,
} from "../lib/apply-inference-status-to-store";
import {
  mergeBackendRecommendedInference,
  resolveFitMaxSeqLength,
  resolveLoadMaxSeqLength,
  resolveManualAutoCtxPin,
} from "../presets/preset-policy";
import { recordLastLocalModelLoad } from "../utils/last-local-model-load";
import {
  ownsModelLoadRun,
  releaseOwnedModelLoadRun,
} from "../utils/model-load-run";
import { ensureGpuDeviceCache } from "@/hooks/use-gpu-info";
import {
  isMultimodalResponse,
} from "../types/api";
import { isExternalModelId } from "../external-providers";
import {
  applyPerModelConfigToRuntime,
  type PerModelConfig,
} from "@/features/model-picker";
import type {
  ChatLoraSummary,
  ChatModelSummary,
} from "../types/runtime";

export type SelectedModelInput = {
  id: string;
  isLora?: boolean;
  ggufVariant?: string;
  /** Where the pick came from (e.g. "hub", "local", "external"). Used to decide
   *  whether an uncached repo should download via the Hub manager. */
  source?: string;
  /** Uncached non-GGUF HF repo staged for a snapshot download (variant null). */
  isHubRepo?: boolean;
  loadingDescription?: string;
  isDownloaded?: boolean;
  expectedBytes?: number;
  forceReload?: boolean;
  nativePathToken?: string;
  nativePathExpiresAtMs?: number | null;
  /** Direct local .gguf file (no HF variant / native token) — still a GGUF
   *  source, so the staging flow treats it as one. */
  isGguf?: boolean;
  throwOnError?: boolean;
  /** Keep the current speculative-decoding choice across the model switch
   *  instead of resetting it to the standing preference. */
  keepSpeculative?: boolean;
  config?: PerModelConfig;
  previousConfig?: PerModelConfig;
};

type LoadingModelState = {
  id: string;
  displayName: string;
  isDownloaded?: boolean;
  isCachedLora?: boolean;
  ggufVariant?: string | null;
  nativePathToken?: string | null;
};

type ActiveModelLoadRun = {
  attemptId: number;
  intentId: number;
  info: LoadingModelState;
  abortController: AbortController;
  completionPromise: Promise<void>;
  resolveCompletion: () => void;
  cancelPromise: Promise<boolean> | null;
  backendLoadStarted: boolean;
  backendLoadModelId: string | null;
  rollbackCheckpoint: string | null;
  rollbackConfig?: PerModelConfig;
  previousCheckpointWasUnloaded: boolean;
};

type SharedModelLoadHandle = {
  run: ActiveModelLoadRun;
  cancel: (preserveCheckpoint?: boolean) => Promise<boolean>;
};

// Multiple surfaces can mount this hook (for example Hub and Chat) while a
// single backend load is shared application-wide. Keep the owning hook's
// cancellation entry point available so another surface can stop that run
// without trying to mutate refs it does not own.
let sharedModelLoadHandle: SharedModelLoadHandle | null = null;

// Approved fingerprints by checkpoint, so a rollback after a failed switch can resend
// the pinned approval the worker requires instead of being blocked.
const approvedRemoteCodeFingerprints = new Map<string, string>();
function rememberApprovedRemoteCode(
  checkpoint: string,
  fingerprint: string | null,
): void {
  if (fingerprint) approvedRemoteCodeFingerprints.set(checkpoint, fingerprint);
}

const MODEL_LOAD_TOAST_CLASSNAMES = {
  toast: "chat-model-load-toast items-center gap-2.5",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
  cancelButton:
    "!h-auto !rounded-none !border-0 !bg-transparent !px-1 !text-ui-11 !font-normal !text-muted-foreground hover:!bg-transparent hover:!text-destructive focus-visible:!text-destructive",
} as const;

const MODEL_LOADED_TOAST_CLASSNAMES = {
  toast: "chat-model-loaded-toast items-center gap-2.5",
} as const;

const LORA_SUFFIX_RE = /_(\d{9,})$/;

function parseTrailingEpoch(input: string): number | undefined {
  const match = input.match(LORA_SUFFIX_RE);
  if (!match) {
    return undefined;
  }
  const parsed = Number.parseInt(match[1], 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function stripTrailingEpoch(input: string): string {
  const cleaned = input.replace(LORA_SUFFIX_RE, "").replace(/[_-]+$/, "").trim();
  return cleaned || input;
}

function shortModelLabel(idOrName: string): string {
  const slash = idOrName.lastIndexOf("/");
  const label = slash >= 0 ? idOrName.slice(slash + 1) : idOrName;
  return label || idOrName;
}

function describeModel(model: {
  is_lora?: boolean;
  is_vision?: boolean;
  is_gguf?: boolean;
  is_mlx?: boolean;
  is_audio?: boolean;
  has_audio_input?: boolean;
}): string | undefined {
  const tags: string[] = [];
  if (model.is_gguf) tags.push("GGUF");
  if (model.is_mlx) tags.push("MLX");
  if (model.is_lora) tags.push("LoRA");
  if (model.is_vision) tags.push("Vision");
  if (model.is_audio) tags.push("Audio");
  if (model.has_audio_input) tags.push("Audio Input");
  if (
    !model.is_lora &&
    !model.is_vision &&
    !model.is_gguf &&
    !model.is_mlx &&
    !model.is_audio &&
    !model.has_audio_input
  )
    tags.push("Base");
  return tags.join(" · ");
}

function toChatModelSummary(model: {
  id: string;
  name?: string | null;
  is_lora?: boolean;
  is_vision?: boolean;
  is_gguf?: boolean;
  is_mlx?: boolean;
  is_audio?: boolean;
  audio_type?: string | null;
  has_audio_input?: boolean;
}): ChatModelSummary {
  return {
    id: model.id,
    name: model.name || model.id,
    description: describeModel(model),
    isLora: Boolean(model.is_lora),
    isVision: Boolean(model.is_vision),
    isGguf: Boolean(model.is_gguf),
    isMlx: Boolean(model.is_mlx),
    isAudio: Boolean(model.is_audio),
    audioType: model.audio_type ?? null,
    hasAudioInput: Boolean(model.has_audio_input),
  };
}

// Merge capability flags from a load/status response into the matching
// models[] entry. /api/models/list omits audio capability for default and
// active-GGUF entries, so the attach gates (`activeModel?.hasAudioInput`)
// would otherwise stay false. Mirrors the compare composer's sync.
// Exported for tests.
export function syncModelCapabilities(
  modelId: string,
  resp: {
    display_name?: string | null;
    is_vision?: boolean;
    is_lora?: boolean;
    is_gguf?: boolean;
    is_audio?: boolean;
    audio_type?: string | null;
    has_audio_input?: boolean;
  },
): void {
  const store = useChatRuntimeStore.getState();
  const models = store.models;
  const synced = {
    isVision: Boolean(resp.is_vision),
    isGguf: Boolean(resp.is_gguf),
    isAudio: Boolean(resp.is_audio),
    audioType: resp.audio_type ?? null,
    hasAudioInput: Boolean(resp.has_audio_input),
  };
  const idx = models.findIndex((m) => m.id === modelId);
  if (idx === -1) {
    store.setModels([
      ...models,
      {
        id: modelId,
        name: resp.display_name || modelId,
        isLora: Boolean(resp.is_lora),
        ...synced,
      },
    ]);
  } else {
    const next = [...models];
    next[idx] = { ...next[idx], ...synced };
    store.setModels(next);
  }
}

function toLoraSummary(lora: {
  display_name: string;
  adapter_path: string;
  base_model?: string | null;
  source?: "training" | "exported" | null;
  export_type?: "lora" | "merged" | "gguf" | null;
}): ChatLoraSummary {
  const idTail = lora.adapter_path.split("/").filter(Boolean).at(-1) ?? "";
  const updatedAt =
    parseTrailingEpoch(lora.display_name) ?? parseTrailingEpoch(idTail);

  return {
    id: lora.adapter_path,
    name: stripTrailingEpoch(lora.display_name),
    baseModel: lora.base_model || "Unknown base model",
    updatedAt,
    source: lora.source ?? undefined,
    exportType: lora.export_type ?? undefined,
  };
}

function getTrustRemoteCodeRequiredMessage(modelName: string): string {
  return `${modelName} was not loaded because its custom code was not approved. Load it again to review the code and approve it.`;
}

function getTransformersUpgradeRequiredMessage(modelName: string): string {
  return `${modelName} was not loaded because it needs a newer transformers release that was not installed. Load it again to install it.`;
}

/**
 * Reconcile the chat runtime store against `/api/inference/status`: refresh the
 * models/loras catalogs and either re-pin the active checkpoint or clear the
 * loaded-model flags when nothing is loaded. Module-level so it can run outside
 * a React render (e.g. the imperative resync below); `useChatModelRuntime.refresh`
 * is a thin wrapper over it. External selections are left untouched since they
 * have no backend mirror.
 */
async function syncInferenceStatusToStore(options?: {
  signal?: AbortSignal;
  includeLoras?: boolean;
}): Promise<void> {
  const signal = options?.signal;
  const includeLoras = options?.includeLoras ?? true;
  const { setModels, setLoras, setCheckpoint, setModelsError } =
    useChatRuntimeStore.getState();
  setModelsError(null);
  try {
    const [listRes, statusRes, lorasRes] = await Promise.all([
      listModels(),
      getInferenceStatus(),
      includeLoras ? listLoras() : Promise.resolve(null),
    ]);

    // Cancellation can land while the requests above are in flight. Bail
    // before writing backend state back -- cancelLoading already cleared it.
    if (signal?.aborted) return;

    setModels(listRes.models.map(toChatModelSummary));
    if (lorasRes) {
      setLoras(lorasRes.loras.map(toLoraSummary));
    }

    const selectedCheckpoint = useChatRuntimeStore.getState().params.checkpoint;
    const isExternalSelectionActive = isExternalModelId(selectedCheckpoint);
    if (statusRes.active_model && !isExternalSelectionActive) {
      const checkpointId = resolveInferenceCheckpointId(statusRes);
      if (checkpointId) {
        const previousGgufVariant =
          useChatRuntimeStore.getState().activeGgufVariant;
        setCheckpoint(checkpointId, statusRes.gguf_variant);
        applyActiveModelStatusToStore(statusRes, {
          previousCheckpoint: selectedCheckpoint,
          previousGgufVariant,
        });
        // setModels(listRes...) above used catalog data, which omits audio
        // capability. Re-apply live status so attach gates survive a refresh.
        syncModelCapabilities(checkpointId, statusRes);
      }
    } else if (!statusRes.active_model && !isExternalSelectionActive) {
      useChatRuntimeStore.setState({
        modelRequiresTrustRemoteCode: false,
        loadedIsMultimodal: false,
        loadedIsDiffusion: false,
      });
    }
  } catch (error) {
    if (signal?.aborted) return;
    const message =
      error instanceof Error ? error.message : "Failed to load models";
    setModelsError(message);
    toast.error("Failed to refresh models", {
      description: message,
    });
  }
}

/**
 * Reconcile the UI after the SERVER unloaded the active model out from under it
 * (e.g. a llama.cpp update unloads the running model to swap the binary): the
 * model selector drops to "select model" instead of pointing at a model that now
 * 400s on send. Imperative so the global llama-update banner (which has no
 * chat-runtime handle) can call it.
 *
 * Only a LOCAL selection points at the unloaded model. An external-provider
 * selection has no llama.cpp mirror and still works, so clearing it (which also
 * wipes its persisted id) would drop a valid, unrelated model; skip the clear so
 * the refresh below leaves it intact.
 */
export async function resyncInferenceStatusAfterServerModelChange(): Promise<void> {
  if (!isExternalModelId(useChatRuntimeStore.getState().params.checkpoint)) {
    useChatRuntimeStore.getState().clearCheckpoint();
  }
  await syncInferenceStatusToStore();
}

function pickOf(info: {
  id: string;
  ggufVariant?: string | null;
  nativePathToken?: string | null;
}): LoadingModelPick {
  return {
    id: info.id,
    ggufVariant: info.ggufVariant ?? null,
    nativePathToken: info.nativePathToken ?? null,
  };
}

export function useChatModelRuntime() {
  const params = useChatRuntimeStore((state) => state.params);
  const models = useChatRuntimeStore((state) => state.models);
  const loras = useChatRuntimeStore((state) => state.loras);
  const setParams = useChatRuntimeStore((state) => state.setParams);
  const setModelsError = useChatRuntimeStore((state) => state.setModelsError);
  const setLastModelLoadError = useChatRuntimeStore(
    (state) => state.setLastModelLoadError,
  );
  const clearCheckpoint = useChatRuntimeStore((state) => state.clearCheckpoint);

  const [loadingModel, setLoadingModel] = useState<LoadingModelState | null>(
    null,
  );
  const [loadToastDismissed, setLoadToastDismissed] = useState(false);
  const [loadProgress, setLoadProgress] = useState<{
    percent: number | null;
    label: string | null;
    phase: "downloading" | "starting";
  } | null>(null);
  const loadAbortRef = useRef<AbortController | null>(null);
  const loadingModelRef = useRef<typeof loadingModel>(null);
  const loadToastIdRef = useRef<string | number | null>(null);
  const loadAttemptRef = useRef(0);
  const loadIntentRef = useRef(0);
  const activeLoadRunRef = useRef<ActiveModelLoadRun | null>(null);
  const loadToastDismissedRef = useRef(false);

  const setLoadToastDismissedState = useCallback((dismissed: boolean) => {
    loadToastDismissedRef.current = dismissed;
    setLoadToastDismissed(dismissed);
  }, []);

  const resetLoadingUi = useCallback((run: ActiveModelLoadRun) => {
    if (!ownsModelLoadRun(activeLoadRunRef.current, run)) return;
    // Cancellation owns the slot until /unload settles. The aborted load's
    // finally block may arrive first, but must not release that slot.
    if (run.cancelPromise) return;
    activeLoadRunRef.current = releaseOwnedModelLoadRun(
      activeLoadRunRef.current,
      run,
    );
    if (sharedModelLoadHandle?.run === run) {
      sharedModelLoadHandle = null;
    }
    setLoadingModel(null);
    setLoadProgress(null);
    loadingModelRef.current = null;
    loadAbortRef.current = null;
    loadToastIdRef.current = null;
    setLoadToastDismissedState(false);
    useChatRuntimeStore.getState().clearLoadingModelPick(pickOf(run.info));
    useChatRuntimeStore.getState().setModelLoading(false);
  }, [setLoadToastDismissedState]);

  const renderLoadDescription = useCallback(
    (
      title: string,
      message: string,
      progressPercent?: number | null,
      progressLabel?: string | null,
    ) =>
      createElement(ModelLoadDescription, {
        title,
        message,
        progressPercent,
        progressLabel,
      }),
    [],
  );

  const refresh = useCallback(
    (options?: { signal?: AbortSignal; includeLoras?: boolean }) =>
      syncInferenceStatusToStore(options),
    [],
  );

  const cancelLoadRun = useCallback(
    async (
      expectedRun?: ActiveModelLoadRun,
      preserveCheckpoint = false,
    ): Promise<boolean> => {
      const run = activeLoadRunRef.current;
      if (!run || (expectedRun && !ownsModelLoadRun(run, expectedRun))) {
        return false;
      }
      if (run.cancelPromise) return run.cancelPromise;

      const model = run.info;
      const backendLoadModelId = run.backendLoadModelId ?? model.id;
      run.abortController.abort();
      useHfTokenWarningStore.getState().resolve("cancel", run);
      useRemoteCodeConsentDialogStore.getState().resolve(false, run);
      useTransformersUpgradeDialogStore.getState().cancelPending(run);
      loadAbortRef.current = null;
      loadingModelRef.current = null;
      const tid = loadToastIdRef.current;
      loadToastIdRef.current = null;
      setLoadingModel(null);
      setLoadProgress(null);
      setLoadToastDismissedState(false);
      if (!preserveCheckpoint) clearCheckpoint();
      if (tid != null) toast.dismiss(tid);
      const isCachedOrLocal = model.isDownloaded || model.isCachedLora;
      toast.info("Stopping model load", {
        description: isCachedOrLocal
          ? undefined
          : "The current download may still finish in the background.",
      });
      useChatRuntimeStore.getState().setModelLoading(true);
      const cancelPromise = (async (): Promise<boolean> => {
        try {
          // AbortController cannot interrupt every preflight await (for
          // example, a consent dialog), and it is not passed to the backend
          // /load request. /unload is the operation that promptly interrupts an
          // in-flight backend load, so send it before waiting for the frontend
          // task to observe the abort. The unload route waits for cancellation
          // to settle before it returns, preventing a late /load activation.
          await unloadModel({ model_path: backendLoadModelId });
          if (
            run.rollbackCheckpoint &&
            (run.backendLoadStarted ||
              run.rollbackCheckpoint.toLowerCase() === model.id.toLowerCase())
          ) {
            run.previousCheckpointWasUnloaded = true;
          }
          // Every owned confirmation above is now declined. If a transformers
          // install is already running, cancelPending deliberately resolves
          // only after it settles, because that install may unload the active
          // backend model. Do not let a replacement start before the cancelled
          // frontend task has observed that outcome.
          await run.completionPromise;
          return true;
        } catch (error) {
          const detail =
            error instanceof Error ? error.message : "Unknown unload error";
          const message = `Failed to stop ${model.displayName}`;
          setModelsError(`${message}: ${detail}`);
          toast.error(message, { description: detail });
          // The request failed, so reconcile against the backend before
          // releasing the UI slot. The caller still receives false and will
          // not start a replacement load from an uncertain backend state.
          try {
            await refresh();
            // refresh() begins by clearing catalog errors. Keep the primary
            // unload failure visible after reconciliation completes.
            setModelsError(`${message}: ${detail}`);
          } catch (refreshError) {
            const refreshDetail =
              refreshError instanceof Error
                ? refreshError.message
                : "Unknown status refresh error";
            setModelsError(
              `${message}: ${detail}. Failed to refresh model status: ${refreshDetail}`,
            );
          }
          return false;
        } finally {
          if (ownsModelLoadRun(activeLoadRunRef.current, run)) {
            activeLoadRunRef.current = releaseOwnedModelLoadRun(
              activeLoadRunRef.current,
              run,
            );
            useChatRuntimeStore
              .getState()
              .clearLoadingModelPick(pickOf(model));
            useChatRuntimeStore.getState().setModelLoading(false);
          }
          if (sharedModelLoadHandle?.run === run) {
            sharedModelLoadHandle = null;
          }
        }
      })();
      run.cancelPromise = cancelPromise;
      return cancelPromise;
    },
    [
      clearCheckpoint,
      refresh,
      setLoadToastDismissedState,
      setModelsError,
    ],
  );

  const cancelLoading = useCallback(
    (preserveCheckpoint = false): Promise<boolean> => {
      const localRun = activeLoadRunRef.current;
      if (localRun) return cancelLoadRun(localRun, preserveCheckpoint);
      const shared = sharedModelLoadHandle;
      return shared ? shared.cancel(preserveCheckpoint) : Promise.resolve(false);
    },
    [cancelLoadRun],
  );

  const invalidatePendingModelSelection = useCallback((): number => {
    loadIntentRef.current += 1;
    return loadIntentRef.current;
  }, []);

  const isModelSelectionIntentCurrent = useCallback((intentId: number) => {
    return loadIntentRef.current === intentId;
  }, []);

  const selectModel = useCallback(
    async (selection: string | SelectedModelInput) => {
      const modelId = typeof selection === "string" ? selection : selection.id;
      const ggufVariant =
        typeof selection === "string" ? undefined : selection.ggufVariant;
      const forceReload =
        typeof selection === "string" ? false : selection.forceReload ?? false;
      const nativePathToken =
        typeof selection === "string" ? undefined : selection.nativePathToken;
      const nativePathExpiresAtMs =
        typeof selection === "string"
          ? null
          : selection.nativePathExpiresAtMs ?? null;
      const explicitIsGguf =
        typeof selection === "string" ? undefined : selection.isGguf;
      const throwOnError =
        typeof selection === "string" ? false : selection.throwOnError ?? false;
      const keepSpeculative =
        typeof selection === "string" ? false : selection.keepSpeculative ?? false;
      let previousConfig =
        typeof selection === "string" ? undefined : selection.previousConfig;
      const currentVariant = useChatRuntimeStore.getState().activeGgufVariant;
      if (!forceReload && (!modelId || (params.checkpoint === modelId && (ggufVariant ?? null) === (currentVariant ?? null)))) {
        if (previousConfig) {
          applyPerModelConfigToRuntime(previousConfig);
        }
        return;
      }
      const initialActiveRun = activeLoadRunRef.current;
      const initialInFlightLoad =
        initialActiveRun?.info ??
        useChatRuntimeStore.getState().loadingModelPick;
      const initiallyLoadingSamePick =
        initialInFlightLoad?.id === modelId &&
        (initialInFlightLoad.ggufVariant ?? null) === (ggufVariant ?? null) &&
        (initialInFlightLoad.nativePathToken ?? null) ===
          (nativePathToken ?? null);
      if (initiallyLoadingSamePick && !initialActiveRun?.cancelPromise) {
        if (previousConfig) {
          applyPerModelConfigToRuntime(previousConfig);
        }
        return;
      }
      // Register the user's intent before awaiting cancellation. This prevents
      // the superseded run's error cleanup from restoring its previous config
      // over settings already applied by the new caller.
      const loadIntentId = ++loadIntentRef.current;
      let replacementNeedsRollback = false;
      // A different pick supersedes the local load in flight. Await its
      // cancellation and backend unload before claiming the slot. Re-check in a
      // loop because multiple selections may be waiting on the same cancellation;
      // after the first resumes and starts a replacement, the last selection wins
      // by cancelling that newer run rather than starting concurrently.
      while (true) {
        const activeRun = activeLoadRunRef.current;
        const inFlightLoad =
          activeRun?.info ??
          useChatRuntimeStore.getState().loadingModelPick;
        if (!inFlightLoad) break;

        const loadingSamePick =
          inFlightLoad.id === modelId &&
          (inFlightLoad.ggufVariant ?? null) === (ggufVariant ?? null) &&
          (inFlightLoad.nativePathToken ?? null) === (nativePathToken ?? null);
        if (loadingSamePick && !activeRun?.cancelPromise) {
          if (previousConfig) {
            applyPerModelConfigToRuntime(previousConfig);
          }
          return;
        }

        if (activeRun?.rollbackConfig) {
          previousConfig = activeRun.rollbackConfig;
        }

        // A load owned by another runtime surface cannot be safely cancelled
        // through this hook's local refs. Delegate to the owning hook through
        // the shared handle instead.
        if (!activeRun) {
          const shared = sharedModelLoadHandle;
          if (shared) {
            if (shared.run.rollbackConfig) {
              previousConfig = shared.run.rollbackConfig;
            }
            const stopped = await shared.cancel(true);
            replacementNeedsRollback =
              replacementNeedsRollback ||
              shared.run.previousCheckpointWasUnloaded;
            if (loadIntentRef.current !== loadIntentId) {
              if (throwOnError) {
                throw new Error("Model selection was superseded by a newer choice.");
              }
              return;
            }
            if (stopped) continue;
          }
          if (previousConfig) {
            applyPerModelConfigToRuntime(previousConfig);
          }
          const message =
            "The current model could not be stopped, so the new model was not loaded.";
          setModelsError(message);
          if (throwOnError) throw new Error(message);
          return;
        }

        // Keep the working checkpoint as the rollback target for the
        // replacement. A standalone Stop still clears the selection.
        const stopped = await cancelLoadRun(activeRun, true);
        replacementNeedsRollback =
          replacementNeedsRollback ||
          activeRun.previousCheckpointWasUnloaded;
        if (loadIntentRef.current !== loadIntentId) {
          if (throwOnError) {
            throw new Error("Model selection was superseded by a newer choice.");
          }
          return;
        }
        if (!stopped) {
          if (previousConfig) {
            applyPerModelConfigToRuntime(previousConfig);
          }
          const message =
            "The current model could not be stopped, so the new model was not loaded.";
          if (throwOnError) throw new Error(message);
          return;
        }
      }

      const explicitIsLora =
        typeof selection === "string" ? undefined : selection.isLora;
      const extraLoadingDescription =
        typeof selection === "string" ? undefined : selection.loadingDescription;
      const isDownloaded =
        typeof selection === "string" ? false : selection.isDownloaded ?? false;
      const model = models.find((entry) => entry.id === modelId);
      const lora = loras.find((entry) => entry.id === modelId);
      // A native path-token selection is a local GGUF by construction (the
      // native model intents only grant .gguf files), but its id is a display
      // label that need not end in ".gguf" -- without this, Manual + Auto
      // layers would pin the UI context instead of letting --fit size it.
      const isGguf =
        explicitIsGguf ??
        (ggufVariant != null ||
          nativePathToken != null ||
          model?.isGguf === true);
      const loraIsAdapter = lora?.exportType === "lora";
      const isLora =
        explicitIsLora ?? model?.isLora ?? loraIsAdapter ?? false;
      const displayName = model?.name || lora?.name || modelId;
      const toastDisplayName = shortModelLabel(displayName);
      const loadAttemptId = ++loadAttemptRef.current;
      primeNativeNotificationPermission().catch(() => undefined);
      const notificationModelKey = `${modelId}:${ggufVariant ?? ""}:${loadAttemptId}`;
      const safeModelName = safeNotificationLabel(toastDisplayName, "The model");
      const currentCheckpoint =
        useChatRuntimeStore.getState().params.checkpoint;
      const previousCheckpoint = currentCheckpoint;
      const previousVariant =
        useChatRuntimeStore.getState().activeGgufVariant ?? null;
      const reloadingSameModel =
        previousCheckpoint === modelId &&
        (ggufVariant ?? null) === (previousVariant ?? null);
      const previousModel = previousCheckpoint
        ? models.find((entry) => entry.id === previousCheckpoint)
        : undefined;
      const previousLora = previousCheckpoint
        ? loras.find((entry) => entry.id === previousCheckpoint)
        : undefined;
      const previousIsLora =
        previousModel?.isLora ?? (previousLora?.exportType === "lora");
      const isLocal = isLocalModelPath(modelId);
      const isCachedLora = isLora && isLocal;
      const loadingDescription = [
        currentCheckpoint ? "Switching models." : null,
        extraLoadingDescription ?? null,
        isDownloaded ? "Loading cached model into memory." : null,
        !isDownloaded && isCachedLora ? "Loading trained model into memory." : null,
      ]
        .filter(Boolean)
        .join(" ");
      setModelsError(null);
      setLastModelLoadError(null); // clear prior failed-load marker
      setLoadToastDismissedState(false);
      const loadInfo = {
        id: modelId,
        displayName,
        isDownloaded,
        isCachedLora,
        ggufVariant: ggufVariant ?? null,
        nativePathToken: nativePathToken ?? null,
      };
      setLoadingModel(loadInfo);
      useChatRuntimeStore.getState().setModelLoading(true);
      useChatRuntimeStore.getState().setLoadingModelPick(pickOf(loadInfo));
      setLoadProgress(
        isDownloaded || isCachedLora
          ? { percent: null, label: null, phase: "starting" }
          : { percent: 0, label: "Preparing download", phase: "downloading" },
      );
      loadingModelRef.current = loadInfo;
      const abortCtrl = new AbortController();
      loadAbortRef.current = abortCtrl;
      let resolveCompletion!: () => void;
      const completionPromise = new Promise<void>((resolve) => {
        resolveCompletion = resolve;
      });
      const run: ActiveModelLoadRun = {
        attemptId: loadAttemptId,
        intentId: loadIntentId,
        info: loadInfo,
        abortController: abortCtrl,
        completionPromise,
        resolveCompletion,
        cancelPromise: null,
        backendLoadStarted: false,
        backendLoadModelId: null,
        rollbackCheckpoint: previousCheckpoint,
        rollbackConfig: previousConfig,
        previousCheckpointWasUnloaded: replacementNeedsRollback,
      };
      activeLoadRunRef.current = run;
      sharedModelLoadHandle = {
        run,
        cancel: (preserveCheckpoint = false) =>
          cancelLoadRun(run, preserveCheckpoint),
      };
      try {
        async function performLoad(): Promise<void> {
          if (abortCtrl.signal.aborted) throw new Error("Cancelled");
          let previousWasUnloaded = run.previousCheckpointWasUnloaded;
          const currentCheckpoint =
            useChatRuntimeStore.getState().params.checkpoint;
          const stateBeforeUnload = useChatRuntimeStore.getState();
          let trustRemoteCode = stateBeforeUnload.params.trustRemoteCode ?? false;
          let approvedRemoteCodeFingerprint: string | null = null;
          const maxSeqLength = stateBeforeUnload.params.maxSeqLength;
          const previousActiveNativePathToken =
            stateBeforeUnload.activeNativePathToken;
          const previousIsGguf =
            previousModel?.isGguf === true
            || previousVariant != null
            || previousActiveNativePathToken != null
            || (previousCheckpoint?.toLowerCase().endsWith(".gguf") ?? false);
          // Roll back to the previous model's own context. previousConfig was
          // snapshotted before this load pre-applied the next model's config, so
          // params.maxSeqLength may already be the next model's; use it only when
          // no snapshot exists.
          const previousMaxSeqLength =
            previousConfig?.maxSeqLength ?? maxSeqLength;
          // Respect the rolled-back model's auto-layers mode: a Manual+Auto model
          // with an unpinned context must reload with 0 (so --fit re-auto-sizes),
          // not the positive context it picked (which the backend treats as a pin).
          const rollbackMaxSeqLength = resolveFitMaxSeqLength(
            previousIsGguf,
            stateBeforeUnload.loadedGpuMemoryMode ?? "auto",
            stateBeforeUnload.loadedGpuLayers ?? GPU_LAYERS_AUTO,
            stateBeforeUnload.loadedCustomContextLength,
            previousIsGguf
              ? (stateBeforeUnload.ggufContextLength ?? 0)
              : previousMaxSeqLength,
          );
          const hfToken = stateBeforeUnload.hfToken || null;
          const previousModelRequiresTrustRemoteCode =
            stateBeforeUnload.modelRequiresTrustRemoteCode;
          const previousActiveNativePathExpiresAtMs =
            stateBeforeUnload.activeNativePathExpiresAtMs;
          // Snapshot the load settings at click time, before the awaits below
          // (validation, the trust dialog, unload).
          const loadChatTemplateOverride = stateBeforeUnload.chatTemplateOverride;
          const loadKvCacheDtype = stateBeforeUnload.kvCacheDtype;
          // gpuMemoryMode is a standing preference (kept across a model switch);
          // the rest are per-model knobs the reset below clears, so they are
          // re-baselined there in lock-step with the store.
          let loadCustomContextLength = stateBeforeUnload.customContextLength;
          const loadGgufContextLength = stateBeforeUnload.ggufContextLength;
          const loadTensorParallel = stateBeforeUnload.tensorParallel;
          const loadActivePresetSource = stateBeforeUnload.activePresetSource;
          const loadActiveGgufVariant = stateBeforeUnload.activeGgufVariant;
          const loadGpuMemoryMode = stateBeforeUnload.gpuMemoryMode;
          let loadGpuLayers = stateBeforeUnload.gpuLayers;
          let loadNCpuMoe = stateBeforeUnload.nCpuMoe;
          let loadSplitRatio = stateBeforeUnload.splitRatio;
          // Reconcile the persisted pick against the GPUs present now, so a stale
          // cross-host / now-hidden pick is dropped before /load rather than
          // rejected there. Warm the device cache first: load-on-selection can
          // run before any GPU hook mounted, and a cold cache would pass the
          // pick through unvalidated. validateGpuIds derives from this too.
          if (stateBeforeUnload.selectedGpuIds != null) {
            await ensureGpuDeviceCache();
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");
          }
          let loadSelectedGpuIds = reconcilePersistedGpuIds(
            stateBeforeUnload.selectedGpuIds,
          );
          let loadSpeculativeType = stateBeforeUnload.speculativeType;
          let loadSpecDraftNMax = stateBeforeUnload.specDraftNMax;
          try {
            // Lightweight pre-flight validation: avoid unloading a working model
            // if the new identifier is clearly invalid (e.g. bad HF id / path).
            const validateNativePathLease = nativePathToken
              ? (await consumeNativePathToken(nativePathToken, "validate-model")).nativePathLease
              : undefined;
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");
            // Validate with the same effective context /load uses: a GGUF native
            // context can exceed maxSeqLength, so sizing on raw maxSeqLength could
            // pass, unload, then have /load refuse it. Uses the click-time
            // snapshot (same values loadModel uses below), so the two agree.
            // Mirror /load on a cross-model switch: the reset below clears the
            // per-model Auto-layers context pin + GPU pick; gpuMemoryMode is a
            // standing preference kept across the switch. A same-repo quant switch
            // (different gguf_variant) is a different model for per-model knobs
            // (context/gpuLayers/pick/MoE are per variant), so re-baseline them too.
            const switchingModelOrVariant =
              currentCheckpoint !== modelId ||
              (loadActiveGgufVariant ?? null) !== (ggufVariant ?? null);
            const resetsPerModelSettings = Boolean(
              currentCheckpoint && switchingModelOrVariant && !keepSpeculative,
            );
            const validateCustomContextLength = resetsPerModelSettings
              ? null
              : loadCustomContextLength;
            const validateGpuIds = resetsPerModelSettings
              ? null
              : loadSelectedGpuIds;
            // The reset below re-baselines gpuLayers to Auto; mirror it here.
            const validateGpuLayers = resetsPerModelSettings
              ? GPU_LAYERS_AUTO
              : loadGpuLayers;
            const validateMaxSeqLength = resolveFitMaxSeqLength(
              isGguf,
              loadGpuMemoryMode,
              validateGpuLayers,
              validateCustomContextLength,
              resolveLoadMaxSeqLength({
                modelId,
                ggufVariant,
                isGguf,
                customContextLength: validateCustomContextLength,
                ggufContextLength: loadGgufContextLength,
                currentCheckpoint,
                activeGgufVariant: loadActiveGgufVariant,
                maxSeqLength,
                presetSource: loadActivePresetSource,
              }),
            );
            const validation = await validateModel({
              model_path: modelId,
              nativePathLease: validateNativePathLease,
              hf_token: hfToken,
              max_seq_length: validateMaxSeqLength,
              load_in_4bit: true,
              is_lora: isLora,
              gguf_variant: ggufVariant ?? null,
              gpu_ids: validateGpuIds ?? undefined,
              ...(isGguf ? { gpu_memory_mode: loadGpuMemoryMode } : {}),
            }, {
              dialogOwner: run,
              signal: abortCtrl.signal,
            });
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");
            // Upgrade consent runs before the security dialogs; Accept installs and the load continues.
            if (validation.requires_transformers_upgrade) {
              const upgraded = await confirmTransformersUpgradeIfNeeded({
                modelName: modelId,
                upgrade: validation.transformers_upgrade,
                // No installable release: custom-code models may fall back to the trust_remote_code gate below.
                trustRemoteCodeFallback: validation.requires_trust_remote_code,
                dialogOwner: run,
              });
              // The install unloads the previous model before the swap (even when
              // the swap then fails), so any exit after this point must roll back.
              // False for the custom-code fallback, which resolves without installing.
              if (
                useTransformersUpgradeDialogStore
                  .getState()
                  .consumeServerUnloadedChat()
                && currentCheckpoint
              ) {
                previousWasUnloaded = true;
                run.previousCheckpointWasUnloaded = true;
              }
              if (!upgraded) {
                throw new Error(getTransformersUpgradeRequiredMessage(displayName));
              }
            }
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");
            // Open the consent dialog when the model needs custom-code consent or has a
            // flagged unsafe file. Fires even when trustRemoteCode is preset on, since the
            // worker requires a matching fingerprint that only the dialog produces.
            if (
              validation.requires_trust_remote_code
              || validation.requires_security_review
            ) {
              const approved = await confirmRemoteCodeIfNeeded({
                modelName: modelId,
                hfToken,
                requiresTrustRemoteCode: true,
                onApprove: (fp) => {
                  trustRemoteCode = true;
                  approvedRemoteCodeFingerprint = fp;
                },
                dialogOwner: run,
                signal: abortCtrl.signal,
              });
              if (!approved) {
                throw new Error(getTrustRemoteCodeRequiredMessage(displayName));
              }
            }
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");
            const loadNativePathLease = nativePathToken
              ? (await consumeNativePathToken(nativePathToken, "load-model")).nativePathLease
              : undefined;
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");

            if (currentCheckpoint && !previousWasUnloaded) {
              await unloadModel({ model_path: currentCheckpoint });
              previousWasUnloaded = true;
              run.previousCheckpointWasUnloaded = true;
            }
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");

            // On a model switch, fall back to the persisted standing
            // preference rather than null so a per-session forced MTP mode
            // can't follow the user onto a model without an MTP head.
            // spec_draft_n_max is MTP-only and always resets. The loaded
            // shadow is seeded too, preventing a transient dirty Apply state.
            // keepSpeculative skips this for a staged Load: the user picked the
            // mode for this model on the sidebar, so honor it (the backend still
            // falls back at runtime if the model has no MTP head).
            if (resetsPerModelSettings) {
              const persistedSpeculativeType = readPersistedSpeculativeType();
              useChatRuntimeStore.setState({
                speculativeType: persistedSpeculativeType,
                loadedSpeculativeType: persistedSpeculativeType,
                specDraftNMax: null,
                loadedSpecDraftNMax: null,
                // Per-model GPU knobs must not follow onto a different model
                // (gpuMemoryMode is a standing preference and is kept).
                selectedGpuIds: null,
                gpuLayers: GPU_LAYERS_AUTO,
                nCpuMoe: 0,
                splitRatio: null,
                // A Manual+Auto context pin is per-model; clear it so a different
                // model loads at Auto/native, not the previous model's pin.
                customContextLength: null,
              });
              loadSpeculativeType = persistedSpeculativeType;
              loadSpecDraftNMax = null;
              // Keep the click-time snapshot in lock-step with the store reset so
              // the load below sizes against the cleared per-model knobs, not the
              // previous model's (gpuMemoryMode is standing, so left as captured).
              loadCustomContextLength = null;
              loadSelectedGpuIds = null;
              loadGpuLayers = GPU_LAYERS_AUTO;
              loadNCpuMoe = 0;
              loadSplitRatio = null;
            }

            // Pinning layers on the SAME model keeps the currently resolved
            // context: with no explicit pin, a manual+pinned reload would send 0,
            // which the backend's --fit off branch treats as the NATIVE context --
            // far larger than the sheet shows when the load was fit-sized (Default
            // or Manual + Auto layers may auto-reduce context to fit VRAM), a
            // likely OOM. ggufContextLength is that resolved value; a model already
            // at native reloads unchanged, so this is safe for any prior mode.
            if (
              isGguf &&
              !switchingModelOrVariant &&
              loadGpuMemoryMode === "manual" &&
              loadGpuLayers >= 0 &&
              loadCustomContextLength == null &&
              (loadGgufContextLength ?? 0) > 0
            ) {
              loadCustomContextLength = loadGgufContextLength;
            }
            const effectiveMaxSeqLength = resolveLoadMaxSeqLength({
              modelId,
              ggufVariant,
              isGguf,
              customContextLength: loadCustomContextLength,
              ggufContextLength: loadGgufContextLength,
              currentCheckpoint,
              activeGgufVariant: loadActiveGgufVariant,
              maxSeqLength,
              presetSource: loadActivePresetSource,
            });
            const loadMaxSeqLength = resolveFitMaxSeqLength(
              isGguf,
              loadGpuMemoryMode,
              loadGpuLayers,
              loadCustomContextLength,
              effectiveMaxSeqLength,
            );
            const effectiveChatTemplateOverride =
              loadChatTemplateOverride?.trim() ? loadChatTemplateOverride : null;
            run.backendLoadStarted = true;
            run.backendLoadModelId = modelId;
            const loadResponse = await loadModel({
              model_path: modelId,
              nativePathLease: loadNativePathLease,
              hf_token: hfToken,
              max_seq_length: loadMaxSeqLength,
              load_in_4bit: true,
              is_lora: isLora,
              gguf_variant: ggufVariant ?? null,
              trust_remote_code: trustRemoteCode,
              approved_remote_code_fingerprint: approvedRemoteCodeFingerprint,
              chat_template_override: effectiveChatTemplateOverride,
              cache_type_kv: loadKvCacheDtype,
              speculative_type: loadSpeculativeType,
              spec_draft_n_max: loadSpecDraftNMax,
              tensor_parallel: loadTensorParallel,
              gpu_memory_mode: loadGpuMemoryMode,
              gpu_layers: loadGpuLayers,
              n_cpu_moe: loadNCpuMoe,
              tensor_split: loadSplitRatio ?? undefined,
              gpu_ids: loadSelectedGpuIds ?? undefined,
            }, {
              dialogOwner: run,
              signal: abortCtrl.signal,
            });

            // If cancelled while loading, don't update UI to show
            // the model as active -- it's being unloaded.
            if (
              abortCtrl.signal.aborted ||
              loadIntentRef.current !== run.intentId
            ) {
              throw new Error("Cancelled");
            }

            // The load applied this spec mode, so persist the user's standing
            // preference now (the requested intent, not the resolved echo;
            // saveSpeculativeType keeps only the universal auto/ngram/off).
            // Skip for a per-model config (keepSpeculative): that choice is
            // model-specific and must not overwrite the global default.
            if (!keepSpeculative) {
              saveSpeculativeType(loadSpeculativeType);
            }
            // Persist the GPU Memory mode only on a successful load (not on
            // dropdown change), so an abandoned selection doesn't stick.
            persistGpuMemoryModeOnLoad(loadResponse, loadGpuMemoryMode);

            const currentParams = useChatRuntimeStore.getState().params;
            setParams(
              mergeBackendRecommendedInference({
                current: currentParams,
                response: loadResponse,
                modelId,
                presetSource: useChatRuntimeStore.getState().activePresetSource,
              }),
            );
            // Qwen3.5/3.6 small models (0.8B, 2B, 4B, 9B) disable thinking by default
            let reasoningDefault = loadResponse.supports_reasoning ?? false;
            if (reasoningDefault) {
              const mid = modelId.toLowerCase();
              if (mid.includes("qwen3.5") || mid.includes("qwen3.6")) {
                const sizeMatch = mid.match(/(\d+\.?\d*)\s*b/);
                if (sizeMatch && parseFloat(sizeMatch[1]) < 9) {
                  reasoningDefault = false;
                }
              }
            }
            const loadedKv = loadResponse.cache_type_kv ?? null;
            const loadedTp = loadResponse.tensor_parallel ?? false;
            const loadedSpec = normalizeSpeculativeType(
              loadResponse.speculative_type,
            );
            const nativeCtx = loadResponse.is_gguf
              ? (loadResponse.context_length ?? 131072)
              : null;
            const reportedMaxCtx = loadResponse.is_gguf
              ? (loadResponse.max_context_length ?? null)
              : null;
            const reportedNativeCtx = loadResponse.is_gguf
              ? (loadResponse.native_context_length ?? null)
              : null;
            // Keep an explicit Manual+Auto context pin (so a later Apply doesn't
            // revert it to Auto) and retain the user's requested context so
            // re-open/re-save keeps the intended override, not the backend's
            // auto-fit context; null stays null.
            const keepCustomCtx = resolveManualAutoCtxPin(
              loadGpuMemoryMode,
              loadGpuLayers,
              loadCustomContextLength,
            );
            const reasoningAlwaysOn = loadResponse.reasoning_always_on ?? false;
            const reasoningStyle = loadResponse.reasoning_style ?? "enable_thinking";
            const supportsReasoning = loadResponse.supports_reasoning ?? false;
            const supportsTools = loadResponse.supports_tools ?? false;
            // GLM-5.2-style models report their own effort levels (e.g.
            // high|max); everything else keeps the default low/medium/high.
            const reasoningEffortLevels =
              loadResponse.reasoning_effort_levels &&
              loadResponse.reasoning_effort_levels.length > 0
                ? (loadResponse.reasoning_effort_levels as ReasoningEffort[])
                : (["low", "medium", "high"] as const);
            const existingReasoningEffort = useChatRuntimeStore.getState().reasoningEffort;
            const clampedReasoningEffort =
              reasoningStyle === "enable_thinking_effort" ||
              reasoningStyle === "reasoning_effort"
                ? clampReasoningEffortToLevels(
                    existingReasoningEffort,
                    reasoningEffortLevels,
                  )
                : clampLocalReasoningEffort(existingReasoningEffort);
            const ggufMaxContextLength = reportedMaxCtx;
            const nextReasoningEnabled = reasoningAlwaysOn
              ? true
              : reloadingSameModel && supportsReasoning
                ? stateBeforeUnload.reasoningEnabled
                : reasoningDefault;
            rememberApprovedRemoteCode(modelId, approvedRemoteCodeFingerprint);
            useChatRuntimeStore.setState({
              ggufContextLength: nativeCtx,
              ggufMaxContextLength,
              ggufNativeContextLength: reportedNativeCtx,
              modelRequiresTrustRemoteCode:
                loadResponse.requires_trust_remote_code ?? false,
              supportsReasoning,
              reasoningAlwaysOn,
              reasoningEnabled: nextReasoningEnabled,
              reasoningStyle,
              supportsReasoningOff: reasoningStyle !== "reasoning_effort",
              reasoningEffortLevels,
              reasoningEffort: clampedReasoningEffort,
              supportsPreserveThinking: loadResponse.supports_preserve_thinking ?? false,
              supportsTools,
              ...(reloadingSameModel && supportsTools
                ? {
                    toolsEnabled: stateBeforeUnload.toolsEnabled,
                    codeToolsEnabled: stateBeforeUnload.codeToolsEnabled,
                  }
                : resolveToolsEnabledOnLoad(supportsTools)),
              kvCacheDtype: loadedKv,
              loadedKvCacheDtype: loadedKv,
              tensorParallel: loadedTp,
              loadedTensorParallel: loadedTp,
              ...loadedGpuMemoryFields(loadResponse),
              speculativeType: loadedSpec,
              loadedSpeculativeType: loadedSpec,
              specDraftNMax: loadResponse.spec_draft_n_max ?? null,
              loadedSpecDraftNMax: loadResponse.spec_draft_n_max ?? null,
              customContextLength: keepCustomCtx,
              loadedCustomContextLength: keepCustomCtx,
              defaultChatTemplate: loadResponse.chat_template ?? null,
              chatTemplateOverride: effectiveChatTemplateOverride,
              loadedChatTemplateOverride: effectiveChatTemplateOverride,
              loadedIsMultimodal: isMultimodalResponse(loadResponse),
              loadedIsDiffusion: loadResponse.is_diffusion ?? false,
              activeNativePathToken: nativePathToken ?? null,
              activeNativePathExpiresAtMs: nativePathToken
                ? nativePathExpiresAtMs
                : null,
            });
            // Unlock attach menus for capabilities the catalog entry lacked.
            syncModelCapabilities(modelId, loadResponse);
            // Qwen3/3.5/3.6: apply thinking-mode-specific params after load
            if (
              modelId.toLowerCase().includes("qwen3") &&
              (loadResponse.supports_reasoning ?? false)
            ) {
              const store = useChatRuntimeStore.getState();
              if (store.activePresetSource === "builtin-default") {
                const mid = modelId.toLowerCase();
                const needsPresencePenalty =
                  mid.includes("qwen3.5") || mid.includes("qwen3.6");
                const p = nextReasoningEnabled
                  ? {
                      temperature: 0.6,
                      topP: 0.95,
                      topK: 20,
                      minP: 0.0,
                      ...(needsPresencePenalty
                        ? { presencePenalty: 1.5 }
                        : {}),
                    }
                  : {
                      temperature: 0.7,
                      topP: 0.8,
                      topK: 20,
                      minP: 0.0,
                      ...(needsPresencePenalty
                        ? { presencePenalty: 1.5 }
                        : {}),
                    };
                store.setParams({ ...store.params, ...p });
              }
            }
            await refresh({ signal: abortCtrl.signal });
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");
            if (
              !isLora &&
              !(loadResponse.is_lora ?? false) &&
              !nativePathToken &&
              !isLocalModelPath(modelId) &&
              !isExternalModelId(modelId)
            ) {
              if (loadResponse.is_gguf || isGguf || ggufVariant) {
                recordLastLocalModelLoad({
                  id: modelId,
                  kind: "gguf",
                  ggufVariant: ggufVariant ?? null,
                });
              } else {
                recordLastLocalModelLoad({ id: modelId, kind: "model" });
              }
            }
          } catch (error) {
            // Skip rollback if user cancelled -- model is already being unloaded.
            if (abortCtrl.signal.aborted) throw error;
            // If we unloaded a previous model and the new load failed, attempt a rollback.
            if (previousWasUnloaded && previousCheckpoint) {
              let rollbackNativePathLease: string | undefined;
              if (previousActiveNativePathToken) {
                try {
                  rollbackNativePathLease = (
                    await consumeNativePathToken(previousActiveNativePathToken, "load-model")
                  ).nativePathLease;
                } catch {
                  throw new Error(
                    "Could not reload the previous local model: please re-select the file.",
                  );
                }
              }
              try {
                run.backendLoadStarted = true;
                run.backendLoadModelId = previousCheckpoint;
                const rollbackResponse = await loadModel({
                  model_path: previousCheckpoint,
                  nativePathLease: rollbackNativePathLease,
                  hf_token: hfToken,
                  max_seq_length: rollbackMaxSeqLength,
                  load_in_4bit: true,
                  is_lora: previousIsLora,
                  gguf_variant: previousVariant,
                  trust_remote_code:
                    previousModelRequiresTrustRemoteCode || trustRemoteCode,
                  // Resend the previous model's pinned approval so restoring it is not re-blocked.
                  approved_remote_code_fingerprint:
                    approvedRemoteCodeFingerprints.get(previousCheckpoint) ?? null,
                  chat_template_override:
                    stateBeforeUnload.loadedChatTemplateOverride,
                  cache_type_kv: stateBeforeUnload.loadedKvCacheDtype,
                  speculative_type:
                    stateBeforeUnload.loadedSpeculativeType,
                  spec_draft_n_max:
                    stateBeforeUnload.loadedSpecDraftNMax,
                  // Restore the previous model in the split mode it was running,
                  // not the default layer split.
                  tensor_parallel: stateBeforeUnload.loadedTensorParallel ?? false,
                  // Restore the previous model's GPU Memory placement, not backend defaults.
                  gpu_memory_mode: stateBeforeUnload.loadedGpuMemoryMode ?? "auto",
                  gpu_layers: stateBeforeUnload.loadedGpuLayers ?? GPU_LAYERS_AUTO,
                  n_cpu_moe: stateBeforeUnload.loadedNCpuMoe ?? 0,
                  tensor_split: stateBeforeUnload.loadedSplitRatio ?? undefined,
                  gpu_ids: stateBeforeUnload.loadedGpuIds ?? undefined,
                }, {
                  dialogOwner: run,
                  signal: abortCtrl.signal,
                });
                if (abortCtrl.signal.aborted) {
                  throw new Error("Cancelled");
                }
                // The rollback restored the previous checkpoint, so a
                // replacement must unload it before starting its own load.
                previousWasUnloaded = false;
                run.previousCheckpointWasUnloaded = false;
                const rollbackSpeculativeType = normalizeSpeculativeType(
                  rollbackResponse.speculative_type,
                );
                useChatRuntimeStore.setState({
                  activeNativePathToken: previousActiveNativePathToken ?? null,
                  // Restore the previous token's lease together with the token so a
                  // rollback never pairs restored token A with failed load B's expiry.
                  activeNativePathExpiresAtMs: previousActiveNativePathToken
                    ? (previousActiveNativePathExpiresAtMs ?? null)
                    : null,
                  // Restore the editable speculative knobs to the rolled-back
                  // model's; the loaded baselines below come from its reload echo.
                  speculativeType: stateBeforeUnload.loadedSpeculativeType ?? null,
                  specDraftNMax: stateBeforeUnload.loadedSpecDraftNMax ?? null,
                  loadedSpeculativeType: rollbackSpeculativeType,
                  loadedSpecDraftNMax:
                    rollbackResponse.spec_draft_n_max ?? null,
                  loadedKvCacheDtype: rollbackResponse.cache_type_kv ?? null,
                  loadedChatTemplateOverride:
                    stateBeforeUnload.loadedChatTemplateOverride,
                  ...loadedGpuMemoryFields(rollbackResponse),
                  tensorParallel: rollbackResponse.tensor_parallel ?? false,
                  loadedTensorParallel:
                    rollbackResponse.tensor_parallel ?? false,
                  customContextLength:
                    stateBeforeUnload.loadedCustomContextLength,
                  loadedCustomContextLength:
                    stateBeforeUnload.loadedCustomContextLength,
                });
                await refresh();
              } catch {
                // Rollback also failed; surface the original load error below.
              }
            }
            throw error;
          }
        }

        const isCachedLoad = isDownloaded || isCachedLora;
        const toastTitle = isCachedLoad ? "Starting model…" : "Downloading model…";
        const modelLoadToastOptions = (description: ReturnType<typeof renderLoadDescription>) => ({
          description,
          duration: Infinity,
          closeButton: true,
          cancel: {
            label: "Cancel",
            onClick: () => {
              void cancelLoadRun(run);
            },
          },
          classNames: MODEL_LOAD_TOAST_CLASSNAMES,
          onDismiss: (dismissedToast: { id: string | number }) => {
            if (
              !ownsModelLoadRun(activeLoadRunRef.current, run) ||
              loadToastIdRef.current !== dismissedToast.id
            ) {
              return;
            }
            setLoadToastDismissedState(true);
          },
        });
        const toastId = toast(
          null,
          modelLoadToastOptions(
            renderLoadDescription(
              toastTitle,
              loadingDescription,
              isCachedLoad ? null : 0,
              isCachedLoad ? null : "Preparing download",
            ),
          ),
        );
        loadToastIdRef.current = toastId;

        // Poll download progress for non-cached models, then (after download
        // or for cached models) poll the llama-server mmap phase so "Starting
        // model..." doesn't look frozen for minutes on large MoE models.
        let progressInterval: ReturnType<typeof setInterval> | null = null;
        const expectedBytes =
          typeof selection !== "string" ? selection.expectedBytes ?? 0 : 0;

        // Rolling window of byte samples for rate/ETA estimation, shared
        // across download + mmap phases so it survives phase flips.
        type Sample = { t: number; b: number };
        const MIN_SAMPLES = 3;
        const MIN_WINDOW = 3_000; // ms
        const MAX_WINDOW = 15_000; // ms
        const dlSamples: Sample[] = [];
        const mmapSamples: Sample[] = [];

        function estimate(
          samples: Sample[],
          bytes: number,
          total: number,
        ): { rate: number; eta: number; stable: boolean } {
          const now = Date.now();
          // Drop samples if the counter reset (e.g. phase flipped).
          if (samples.length > 0 && bytes < samples[samples.length - 1].b) {
            samples.length = 0;
          }
          samples.push({ t: now, b: bytes });
          const cutoff = now - MAX_WINDOW;
          while (samples.length > 2 && samples[0].t < cutoff) {
            samples.shift();
          }
          if (samples.length < MIN_SAMPLES) {
            return { rate: 0, eta: 0, stable: false };
          }
          const first = samples[0];
          const last = samples[samples.length - 1];
          const dt = (last.t - first.t) / 1000;
          const db = last.b - first.b;
          if (dt * 1000 < MIN_WINDOW || db <= 0) {
            return { rate: 0, eta: 0, stable: false };
          }
          const rate = db / dt;
          const eta =
            total > 0 && bytes < total && rate > 0 ? (total - bytes) / rate : 0;
          return { rate, eta, stable: true };
        }

        function composeProgressLabel(
          dlGb: number,
          totalGb: number,
          bytes: number,
          total: number,
          samples: Sample[],
        ): string {
          const base =
            totalGb > 0
              ? `${dlGb.toFixed(1)} of ${totalGb.toFixed(1)} GB`
              : `${dlGb.toFixed(1)} GB downloaded`;
          const est = estimate(samples, bytes, total);
          if (!est.stable) return base;
          const rateStr = formatRate(est.rate);
          const etaStr = total > 0 ? formatEta(est.eta) : "";
          return etaStr && etaStr !== "--"
            ? `${base} • ${rateStr} • ${etaStr} left`
            : `${base} • ${rateStr}`;
        }

        let downloadComplete = isDownloaded || isCachedLora;

        const pollDownload = async () => {
          if (
            abortCtrl.signal.aborted ||
            !ownsModelLoadRun(activeLoadRunRef.current, run)
          ) {
            if (progressInterval) clearInterval(progressInterval);
            return;
          }
          try {
            const prog =
              ggufVariant && expectedBytes > 0
                ? await getGgufDownloadProgress(modelId, ggufVariant, expectedBytes)
                : await getDownloadProgress(modelId);
            if (
              abortCtrl.signal.aborted ||
              !ownsModelLoadRun(activeLoadRunRef.current, run)
            ) {
              return;
            }

            if (prog.progress > 0 && prog.progress < 1) {
              hasShownProgress = true;
              const dlGb = prog.downloaded_bytes / 1e9;
              const totalGb = prog.expected_bytes / 1e9;
              const pct = Math.round(prog.progress * 100);
              const progressLabel = composeProgressLabel(
                dlGb,
                totalGb,
                prog.downloaded_bytes,
                prog.expected_bytes,
                dlSamples,
              );
              setLoadProgress({
                percent: pct,
                label: progressLabel,
                phase: "downloading",
              });
              if (loadToastDismissedRef.current) return;
              toast(null, {
                id: toastId,
                ...modelLoadToastOptions(
                  renderLoadDescription(
                    "Downloading model…",
                    loadingDescription,
                    pct,
                    progressLabel,
                  ),
                ),
              });
            } else if (
              prog.downloaded_bytes > 0 &&
              prog.expected_bytes === 0 &&
              prog.progress === 0
            ) {
              hasShownProgress = true;
              const dlGb = prog.downloaded_bytes / 1e9;
              const est = estimate(dlSamples, prog.downloaded_bytes, 0);
              const rateSuffix =
                est.stable ? ` • ${formatRate(est.rate)}` : "";
              setLoadProgress({
                percent: null,
                label: `${dlGb.toFixed(1)} GB downloaded${rateSuffix}`,
                phase: "downloading",
              });
            } else if (prog.progress >= 1 && hasShownProgress) {
              downloadComplete = true;
              setLoadProgress({
                percent: 100,
                label: "Download complete",
                phase: "starting",
              });
              if (!loadToastDismissedRef.current) {
                toast(null, {
                  id: toastId,
                  ...modelLoadToastOptions(
                    renderLoadDescription(
                      "Starting model…",
                      "Download complete. Loading the model into memory.",
                      100,
                      "Download complete",
                    ),
                  ),
                });
              }
              notifyNative({
                key: `model-downloaded:${notificationModelKey}`,
                title: "Model downloaded",
                body: `${safeModelName} finished downloading and is loading into memory.`,
                requestPermission: false,
              }).catch(() => undefined);
              // Keep polling: the mmap branch below takes over from here.
            }
          } catch {
            // Ignore polling errors; keep polling.
          }
        };

        const pollLoad = async () => {
          if (
            abortCtrl.signal.aborted ||
            !ownsModelLoadRun(activeLoadRunRef.current, run)
          ) {
            if (progressInterval) clearInterval(progressInterval);
            return;
          }
          try {
            const prog = await getLoadProgress();
            if (
              abortCtrl.signal.aborted ||
              !ownsModelLoadRun(activeLoadRunRef.current, run)
            ) {
              return;
            }
            if (!prog || prog.phase == null) return;
            if (prog.phase === "ready") {
              // Loaded. The chat flow will flip loadingModelRef shortly;
              // just stop polling.
              if (progressInterval) clearInterval(progressInterval);
              return;
            }
            if (prog.bytes_total <= 0) return; // nothing useful to render
            // Decimal GB (1e9) so the total matches the file size Hugging Face
            // reports and the model-picker shows, not the smaller base-1024 GiB.
            const loadedGb = prog.bytes_loaded / 1e9;
            const totalGb = prog.bytes_total / 1e9;
            const pct = Math.min(99, Math.round(prog.fraction * 100));
            const est = estimate(mmapSamples, prog.bytes_loaded, prog.bytes_total);
            const base = `${loadedGb.toFixed(1)} of ${totalGb.toFixed(1)} GB in memory`;
            const label = est.stable
              ? `${base} • ${formatRate(est.rate)}${
                  formatEta(est.eta) !== "--" ? ` • ${formatEta(est.eta)} left` : ""
                }`
              : base;
            setLoadProgress({
              percent: pct,
              label,
              phase: "starting",
            });
            if (loadToastDismissedRef.current) return;
            toast(null, {
              id: toastId,
              ...modelLoadToastOptions(
                renderLoadDescription(
                  "Starting model…",
                  "Paging weights into memory.",
                  pct,
                  label,
                ),
              ),
            });
          } catch {
            // Ignore polling errors.
          }
        };

        const pollProgress = async () => {
          if (!downloadComplete) {
            await pollDownload();
          } else {
            await pollLoad();
          }
        };

        let hasShownProgress = false;
        setTimeout(pollProgress, 500);
        progressInterval = setInterval(pollProgress, 2000);

        try {
          await performLoad();
          // User cancelled mid-refresh; cancelLoading handles teardown.
          if (abortCtrl.signal.aborted) return;
          if (loadToastDismissedRef.current) {
            toast.success(`${toastDisplayName} loaded`, {
              classNames: MODEL_LOADED_TOAST_CLASSNAMES,
              closeButton: true,
              duration: 8000,
            });
          } else {
            toast.success(`${toastDisplayName} loaded`, {
              id: toastId,
              description: undefined,
              cancel: undefined,
              classNames: MODEL_LOADED_TOAST_CLASSNAMES,
              closeButton: true,
              duration: 8000,
              onDismiss: undefined,
            });
          }
          notifyNative({
            key: `model-loaded:${notificationModelKey}`,
            title: "Model ready",
            body: `${safeModelName} is loaded and ready to chat.`,
            requestPermission: false,
          }).catch(() => undefined);
        } catch (err) {
          if (
            !abortCtrl.signal.aborted &&
            ownsModelLoadRun(activeLoadRunRef.current, run)
          ) {
            const message =
              err instanceof Error ? err.message : "Failed to load model";
            if (loadToastDismissedRef.current) {
              toast.error(message);
            } else {
              toast.error(message, {
                id: toastId,
                description: undefined,
                cancel: undefined,
                classNames: undefined,
                closeButton: true,
                duration: 8000,
                onDismiss: undefined,
              });
            }
            notifyNative({
              key: `model-load-failed:${notificationModelKey}`,
              title: "Model failed to load",
              body: sanitizeNotificationBody(message, "The model failed to load."),
              requestPermission: false,
            }).catch(() => undefined);
          }
          throw err;
        } finally {
          if (progressInterval) clearInterval(progressInterval);
          resetLoadingUi(run);
        }
      } catch (error) {
        const isLatestRun =
          loadAttemptRef.current === run.attemptId &&
          loadIntentRef.current === run.intentId;
        if (
          isLatestRun &&
          previousConfig
        ) {
          applyPerModelConfigToRuntime(previousConfig);
        }
        if (abortCtrl.signal.aborted) return false; // User cancelled, nothing to report
        if (!isLatestRun) return false;
        resetLoadingUi(run);
        const message =
          error instanceof Error ? error.message : "Failed to load model";
        setModelsError(message);
        setLastModelLoadError(message); // load-specific failure for the attach gates
        if (throwOnError) {
          throw error instanceof Error ? error : new Error(message);
        }
        return false;
      } finally {
        run.resolveCompletion();
      }
      return true;
    },
    [
      cancelLoadRun,
      loras,
      models,
      params.checkpoint,
      refresh,
      renderLoadDescription,
      resetLoadingUi,
      setLoadToastDismissedState,
      setModelsError,
      setLastModelLoadError,
      setParams,
    ],
  );

  const ejectModel = useCallback(async (): Promise<boolean> => {
    if (!params.checkpoint) {
      return false;
    }
    const runtime = useChatRuntimeStore.getState();
    if (runtime.modelLoading || runtime.loadingModelPick) {
      toast.info("A model is loading", {
        description: "Wait for it to finish or cancel it first.",
      });
      return false;
    }
    setModelsError(null);
    if (isExternalModelId(params.checkpoint)) {
      clearCheckpoint();
      await refresh();
      return true;
    }
    try {
      async function performUnload(): Promise<void> {
        await unloadModel({ model_path: params.checkpoint });
        clearCheckpoint();
        await refresh();
      }

      const unloadPromise = performUnload();
      toast.promise(unloadPromise, {
        loading: "Unloading model",
        success: { message: "Model unloaded", duration: 1200 },
        error: (err) =>
          err instanceof Error ? err.message : "Failed to unload model",
        description: "Releases VRAM and resets inference state.",
      });
      await unloadPromise;
      return true;
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to unload model";
      setModelsError(message);
      return false;
    }
  }, [clearCheckpoint, params.checkpoint, refresh, setModelsError]);

  return {
    refresh,
    selectModel,
    ejectModel,
    cancelLoading,
    invalidatePendingModelSelection,
    isModelSelectionIntentCurrent,
    loadingModel,
    loadProgress,
    loadToastDismissed,
  };
}
