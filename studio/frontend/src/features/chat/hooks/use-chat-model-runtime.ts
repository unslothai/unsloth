// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CACHE_MISS_DOWNLOAD_DESCRIPTION,
  EMPTY_CACHE_MISS_WATCH,
  watchCacheMissDownload,
} from "../lib/cache-miss-download";
import { mlxRuntimeStateFrom } from "../lib/mlx-runtime-state";
import { shouldRestorePreviousModel } from "../lib/restore-previous-model";
import {
  type ServerTuningValues,
  clearedServerTuningState,
  committedServerTuningState,
  serverTuningLoadPayload,
} from "../lib/server-tuning-fields";
import { createElement, useCallback, useEffect, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import {
  isBackendDownForDesktopUpdate,
  isSilencedDesktopUpdateFailure,
} from "@/lib/desktop-update-activity";
import { subscribeModelLifecycle } from "@/lib/model-lifecycle-events";
import {
  type TransferSample,
  appendSample,
  computeTransferStats,
} from "@/lib/transfer-stats";
import { confirmRemoteCodeIfNeeded } from "@/features/security";
import { defaultInferenceParams } from "../presets/preset-policy";
import {
  type ReloadHint,
  serverWideReloadRequired,
} from "../lib/server-wide-reload";
import {
  type OffloadCounts,
  offloadCountsFrom,
  offloadWarning,
} from "../lib/partial-offload";
import { isSettingsRouteAbsent } from "@/features/settings/api/settings-route-absent";
import {
  failureLogPath,
  loadFailureLogFamily,
  viewLogsAction,
} from "@/features/settings/lib/view-logs-action";
import { loadModelMemorySettings } from "@/features/settings/api/model-memory";
import { loadVramBudgetSettings } from "@/features/settings/api/vram-budget";
import {
  listOpenAIModels,
  loadMultiModelEnabled,
  loadOpenAIAutoSwitchSettings,
} from "@/features/settings";
import {
  confirmTransformersUpgradeIfNeeded,
  useTransformersUpgradeDialogStore,
} from "@/features/transformers-upgrade";
import { consumeNativePathToken } from "@/features/native-intents/api";
// eslint-disable-next-line no-restricted-imports -- Avoid the hub barrel's React and download-manager exports.
import {
  isOllamaModelId,
  modelDisplayName,
} from "@/features/hub/lib/model-identity";
// eslint-disable-next-line no-restricted-imports -- Avoid the hub barrel's React and download-manager exports.
import { subscribeResidentStatusRefresh } from "@/features/hub/lib/resident-status-refresh";
import {
  beginServerModelWait,
  serverModelWaitOutstanding,
  statusPollSignal,
} from "../lib/server-model-wait";
// eslint-disable-next-line no-restricted-imports -- The hub barrel imports chat; this lifecycle leaf does not.
import { dismissStartToastsForModelSelection } from "@/features/hub/download-manager";
import { prepareHfTokenForUse } from "@/features/hf-auth";
import { ModelLoadDescription } from "../components/model-load-status";
import {
  getDownloadProgress,
  getGgufDownloadProgress,
  getInferenceStatus,
  getLoadProgress,
  fetchGgufStagedMetadata,
  listLoras,
  listModels,
  loadModel,
  unloadModel,
  validateModel,
} from "../api/chat-api";
import { formatEta, formatRate } from "../utils/format-transfer";
import {
  ownsModelLoadRun,
  releaseOwnedModelLoadRun,
} from "../utils/model-load-run";
import {
  confirmStopRunningChatsIfNeeded,
  type StopRunningChatsDecision,
} from "../utils/confirm-stop-running-chats";
import {
  requestLocalPromptQueueStop,
  requestPromptQueueStop,
  notifyLocalPromptQueueLoadFailed,
} from "../utils/prompt-queue-boundary";
import { cancelPreStreamRunReservations } from "../utils/pre-stream-run-reservation";
import {
  chatModelLifecycleGate,
  type ModelLifecycleLease,
} from "../utils/model-lifecycle-gate";
import {
  GPU_LAYERS_AUTO,
  isLocalModelPath,
  loadedGpuMemoryFields,
  noteLoadedModelReasoningMode,
  persistGpuMemoryModeOnLoad,
  pinHoldsLiveEffort,
  readPersistedGpuMemoryMode,
  readPersistedSpeculativeType,
  reconcilePersistedGpuIds,
  resolvePreserveThinkingOnLoad,
  resolveToolsEnabledOnLoad,
  saveSpeculativeType,
  takeEffortDisplacedByPin,
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
  tryAdoptServerActiveModel,
} from "../lib/apply-inference-status-to-store";
import {
  isIdleUnloadedStatus,
  isSpeechOnlyStatus,
} from "../lib/speech-only-status";
import {
  residentRuntimeMatchesConfig,
  residentSpeculativeNeedsRepair,
} from "../lib/resident-config-match";
import { residentModelMatchesPick } from "../lib/resident-model-match";
import {
  loadedContextForParams,
  mergeBackendRecommendedInference,
  resolveFitMaxSeqLength,
  isReplayedLoadContext,
  unpinnedDefaultRequest,
  unpinnedLoadContext,
  resolveLoadMaxSeqLength,
  resolveExplicitCtxPin,
  loadRequestContextPin,
  replayMaxTokensCap,
} from "../presets/preset-policy";
import { recordLastLocalModelLoad } from "../utils/last-local-model-load";
import { loadFallbackNotice } from "../utils/mmproj-fallback";
import { resolveQwenThinkingParams } from "../utils/qwen-sampling-table";
import { refreshContextUsage } from "../utils/refresh-context-usage";
import { defaultEngineGpuIds, ensureGpuDeviceCache } from "@/hooks/use-gpu-info";
import {
  type CpuFallbackReason,
  type MmprojFallbackReason,
  type InferenceStatusResponse,
  isMultimodalResponse,
} from "../types/api";
import { isExternalModelId } from "../external-providers";
import {
  DEFAULT_MAX_SEQ_LENGTH,
  DEFAULT_PER_MODEL_CONFIG,
  applyPerModelConfigToRuntime,
  currentRuntimePerModelConfig,
  isServedByMlx,
  normalizeMaxSeqLength,
  residentIsServedByMlx,
  resolveInitialConfig,
  savedContextPin,
  type PerModelConfig,
  loadedContextFields,
} from "@/features/model-picker";
import {
  invalidateLlamaFlagCatalog,
  loadManagedLlamaFlags,
} from "@/features/model-picker/api/llama-flags";
import { usePlatformStore } from "@/config/env";
import type {
  ChatLoraSummary,
  ChatModelRow,
} from "../types/runtime";

type LoadingModelState = {
  id: string;
  displayName: string;
  isDownloaded?: boolean;
  isCachedLora?: boolean;
  ggufVariant?: string | null;
  nativePathToken?: string | null;
};

/** The single shared load slot; its token lets late callbacks prove ownership. */
type ActiveModelLoadRun = {
  attemptId: number;
  intentId: number;
  abortController: AbortController;
  /** Sent as load_request_id; /unload's cancel_load_request_id binds to this attempt only. */
  requestId: string;
  /** The model_path POSTed to /load; /unload must name the same target to match. */
  loadAttemptPath: string | null;
  cancelPromise: Promise<boolean> | null;
  rollbackCheckpoint: string | null;
  /** clearCheckpoint() drops activeGgufVariant, so the rollback must carry the variant. */
  rollbackVariant: string | null;
  rollbackConfig?: PerModelConfig;
  rollbackLoadId: string | null;
  rollbackNativePathToken: string | null;
  rollbackNativePathExpiresAtMs: number | null;
  rollbackLoadedState: ReturnType<typeof useChatRuntimeStore.getState>;
  /** Set once the preliminary unload removed the prior resident; cancel must then reconcile. */
  residentModelUnloaded: boolean;
  forceCancelActive: boolean;
  /** Resolves when the owning coroutine unwinds; cancellation holds the slot until then. */
  settledPromise: Promise<void>;
  markSettled: () => void;
};

let modelSelectionIntentEpoch = 0;
let pendingExternalReplacement:
  | { intentId: number; config?: PerModelConfig }
  | null = null;
export type SelectedModelInput = {
  id: string;
  loadId?: string | null;
  isLora?: boolean;
  ggufVariant?: string;
  source?: string;
  isHubRepo?: boolean;
  loadingDescription?: string;
  isDownloaded?: boolean;
  expectedBytes?: number;
  downloadPresentation?: {
    label: string;
    filename: string;
    expectedBytes: number;
  };
  forceReload?: boolean;
  nativePathToken?: string;
  nativePathExpiresAtMs?: number | null;
  isGguf?: boolean;
  /** Undefined means unknown, not text-only. */
  isVision?: boolean;
  isDiffusion?: boolean;
  throwOnError?: boolean;
  keepSpeculative?: boolean;
  config?: PerModelConfig;
  previousConfig?: PerModelConfig;
};

/** An absent route is not a failed read: only the second one withholds an answer. */
async function readReloadHint(
  read: () => Promise<{ reloadRequired: boolean } | null>,
): Promise<ReloadHint> {
  try {
    return (await read()) ?? "unsupported";
  } catch (error) {
    return isSettingsRouteAbsent(error) ? "unsupported" : "unknown";
  }
}

async function readServerWideReloadHints(): Promise<boolean> {
  const [modelMemory, vramBudget] = await Promise.all([
    // Forced: a read started before a save would answer about the replaced policy.
    readReloadHint(() => loadModelMemorySettings({ force: true })),
    // Rethrow: this reader answers null for both a failed read and an absent route.
    readReloadHint(() =>
      loadVramBudgetSettings({ force: true, rethrow: true }),
    ),
  ]);
  return serverWideReloadRequired({ modelMemory, vramBudget });
}

/** Order matters: position decides which card gets the model first. */
function sameGpuSelection(
  left: readonly number[] | null | undefined,
  right: readonly number[] | null | undefined,
): boolean {
  const a = left ?? [];
  const b = right ?? [];
  return a.length === b.length && a.every((id, index) => id === b[index]);
}

/** Restore the rollback snapshot first, or clearCheckpoint persists the cancelled target's config. */
function restoreRollbackConfigForClear(run: ActiveModelLoadRun): void {
  if (!run.rollbackConfig) return;
  applyPerModelConfigToRuntime(run.rollbackConfig, {
    isDiffusion: useChatRuntimeStore.getState().loadedIsDiffusion,
  });
}

const approvedRemoteCodeFingerprints = new Map<string, string>();
type PendingReplacementRollback = {
  checkpoint: string | null;
  config?: PerModelConfig;
  /** The cancelled run unloaded the resident before POSTing /load; replacement must reload it. */
  residentUnloaded?: boolean;
  variant?: string | null;
  loadId?: string | null;
  nativePathToken?: string | null;
  nativePathExpiresAtMs?: number | null;
  loadedState?: ReturnType<typeof useChatRuntimeStore.getState>;
};

// Retained until the winning intent reserves its run: cancel can finish after a newer pick starts.
let pendingReplacementRollback: PendingReplacementRollback | null = null;

const PREFLIGHT_LEASE_RETRY_MS = 250;
function rememberApprovedRemoteCode(
  checkpoint: string,
  fingerprint: string | null,
): void {
  if (fingerprint) approvedRemoteCodeFingerprints.set(checkpoint, fingerprint);
}

const MODEL_LOAD_TOAST_CLASSNAMES = {
  toast: "chat-model-load-toast",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
  cancelButton:
    "!h-auto !rounded-none !border-0 !bg-transparent !px-1 !text-ui-11 !font-normal !text-muted-foreground hover:!bg-transparent hover:!text-destructive focus-visible:!text-destructive",
} as const;

const LORA_SUFFIX_RE = /_(\d{9,})$/;

const CLI_LOAD_POLL_IDLE_MS = 60_000;
const CLI_LOAD_POLL_MAX_MS = 600_000;

async function waitForServerModel(signal?: AbortSignal): Promise<void> {
  const started = Date.now();
  let sawLoad = false;

  while (
    !signal?.aborted &&
    !useChatRuntimeStore.getState().params.checkpoint &&
    !useChatRuntimeStore.getState().modelLoading
  ) {
    let status: InferenceStatusResponse | null = null;
    // Capped, or a half-open read parks the loop past the abort with the gate still up.
    const poll = statusPollSignal(signal);
    try {
      status = await getInferenceStatus(poll.signal);
    } catch {
      // A later poll can recover from a transient status failure.
    } finally {
      poll.dispose();
    }
    if (
      signal?.aborted ||
      useChatRuntimeStore.getState().params.checkpoint ||
      useChatRuntimeStore.getState().modelLoading
    ) {
      return;
    }

    if (status) {
      const loading = (status.loading?.length ?? 0) > 0;
      sawLoad ||= loading;
      if (!loading && status.active_model) {
        await tryAdoptServerActiveModel({ status });
        return;
      }
      if (!loading && sawLoad) return;
    }
    const elapsed = Date.now() - started;
    if (!sawLoad && elapsed >= CLI_LOAD_POLL_IDLE_MS) return;
    if (elapsed >= CLI_LOAD_POLL_MAX_MS) return;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

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
  has_video_input?: boolean;
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

function toChatModelRow(model: {
  id: string;
  name?: string | null;
  is_lora?: boolean;
  is_vision?: boolean;
  is_gguf?: boolean;
  is_mlx?: boolean;
  is_audio?: boolean;
  audio_type?: string | null;
  has_audio_input?: boolean;
  has_video_input?: boolean;
}): ChatModelRow {
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
    hasVideoInput: Boolean(model.has_video_input),
  };
}

// /api/models/list omits audio capability for some entries; merge it from load/status.
export function syncModelCapabilities(
  modelId: string,
  resp: {
    display_name?: string | null;
    is_vision?: boolean;
    is_lora?: boolean;
    is_gguf?: boolean;
    is_mlx?: boolean;
    is_audio?: boolean;
    audio_type?: string | null;
    has_audio_input?: boolean;
    has_video_input?: boolean;
  },
): void {
  const store = useChatRuntimeStore.getState();
  const models = store.models;
  const synced = {
    isVision: Boolean(resp.is_vision),
    isGguf: Boolean(resp.is_gguf),
    isMlx: Boolean(resp.is_mlx),
    isAudio: Boolean(resp.is_audio),
    audioType: resp.audio_type ?? null,
    hasAudioInput: Boolean(resp.has_audio_input),
    // /api/models/list omits this for the active GGUF row.
    hasVideoInput: Boolean(resp.has_video_input),
  };
  const idx = models.findIndex((m) => m.id === modelId);
  if (idx === -1) {
    store.setModels([
      ...models,
      {
        id: modelId,
        name: modelDisplayName(resp.display_name || modelId),
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
  size_bytes?: number | null;
  audio_type?: string | null;
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
    sizeBytes: lora.size_bytes ?? null,
    audioType: lora.audio_type ?? null,
  };
}

function getTrustRemoteCodeRequiredMessage(modelName: string): string {
  return `${modelName} was not loaded because its custom code was not approved. Load it again to review the code and approve it.`;
}

function getTransformersUpgradeRequiredMessage(modelName: string): string {
  return `${modelName} was not loaded because it needs a newer transformers release that was not installed. Load it again to install it.`;
}

// Prevent older concurrent status reads from overwriting newer results.
let syncGeneration = 0;
let loraSyncGeneration = 0;
let lastIdleUnloadArmed = false;

async function readIdleUnloadArmed(): Promise<boolean> {
  try {
    const settings = await loadOpenAIAutoSwitchSettings();
    lastIdleUnloadArmed = settings.idleUnloadActive;
  } catch {
    // Preserve the last answer: treating a settings blip as disarmed would discard the pick.
  }
  return lastIdleUnloadArmed;
}

// A lookup started before a newer status must not write the replaced quant.
let quantLookupGeneration = 0;

function publishLoadedModels(
  ids: string[],
  statusId?: string | null,
  statusQuant?: string | null,
  checkpoints?: string[],
): void {
  const current = useChatRuntimeStore.getState().loadedModels;
  const known = new Map(current.map((m) => [m.checkpoint ?? m.id, m]));
  const next = ids.map((id, i) => {
    const checkpoint = checkpoints?.[i] || id;
    const prev = known.get(checkpoint);
    const quant = id === statusId ? (statusQuant ?? null) : prev?.quant;
    return prev && prev.quant === quant && prev.checkpoint === checkpoint
      ? prev
      : { id, quant, checkpoint };
  });
  if (next.length !== current.length || next.some((m, i) => m !== current[i])) {
    useChatRuntimeStore.setState({ loadedModels: next });
  }
  if (next.every((m) => m.quant !== undefined)) return;
  const lookup = ++quantLookupGeneration;
  void listOpenAIModels().then(
    (models) => {
      if (lookup !== quantLookupGeneration) return;
      const details = new Map(models.filter((m) => m.loaded).map((m) => [m.id, m]));
      const { loadedModels } = useChatRuntimeStore.getState();
      if (!loadedModels.some((m) => m.quant === undefined && details.has(m.id))) return;
      useChatRuntimeStore.setState({
        loadedModels: loadedModels.map((m) =>
          m.quant !== undefined || !details.has(m.id)
            ? m
            : { ...m, quant: details.get(m.id)?.quant ?? null },
        ),
      });
    },
    () => {},
  );
}

function stopQueuedRuns(decision: StopRunningChatsDecision, scoped: boolean): void {
  if (scoped) {
    requestPromptQueueStop(decision.promptQueueThreadIds);
    return;
  }
  cancelPreStreamRunReservations(decision.preStreamRunTokens);
  requestLocalPromptQueueStop(decision.promptQueueThreadIds);
}

function unloadKeptModel(keptId: string): Promise<boolean> {
  return confirmStopRunningChatsIfNeeded("Unloading this model", "unload", keptId).then(
    async (decision) => {
      if (!decision.proceed) return false;
      stopQueuedRuns(decision, true);
      await unloadModel({ model_path: keptId, force_cancel_active: decision.forceCancelActive });
      return true;
    },
  );
}

async function syncInferenceStatusToStore(options?: {
  signal?: AbortSignal;
  includeLoras?: boolean;
  preserveIdleUnloaded?: boolean;
  externalChatSlotLoad?: boolean;
}): Promise<void> {
  const signal = options?.signal;
  const downWhenIssued = isBackendDownForDesktopUpdate();
  const includeLoras = options?.includeLoras ?? true;
  const generation = ++syncGeneration;
  const loraGeneration = includeLoras ? ++loraSyncGeneration : null;
  const superseded = () => generation !== syncGeneration;
  const loraSuperseded = () =>
    loraGeneration !== null && loraGeneration !== loraSyncGeneration;
  const { setModels, setLoras, setCheckpoint, setModelsError } =
    useChatRuntimeStore.getState();
  setModelsError(null);
  try {
    const selectedAtStart = useChatRuntimeStore.getState().params.checkpoint;
    const [listRes, statusRes, , idleUnloadArmed] = await Promise.all([
      listModels(),
      getInferenceStatus(signal, selectedAtStart),
      // Settled from this request alone: a sibling rejection must not mark the inventory settled.
      includeLoras
        ? listLoras().then(
            (lorasRes) => {
              if (!signal?.aborted && !loraSuperseded()) {
                setLoras(lorasRes.loras.map(toLoraSummary));
              }
              return lorasRes;
            },
            (error) => {
              if (!signal?.aborted && !loraSuperseded()) {
                useChatRuntimeStore.setState({ loraInventorySettled: true });
              }
              throw error;
            },
          )
        : Promise.resolve(null),
      options?.preserveIdleUnloaded
        ? readIdleUnloadArmed()
        : Promise.resolve(false),
    ]);

    if (signal?.aborted || superseded()) return;

    setModels(listRes.models.map(toChatModelRow));
    publishLoadedModels(
      statusRes.serving ?? [],
      statusRes.active_model,
      statusRes.gguf_variant,
      statusRes.serving_checkpoints,
    );

    const statusLoading = (statusRes.loading?.length ?? 0) > 0;
    // Adopting the outgoing model would set the checkpoint the observer needs empty.
    if (statusLoading && serverModelWaitOutstanding()) return;

    const selectedCheckpoint = useChatRuntimeStore.getState().params.checkpoint;
    const isExternalSelectionActive = isExternalModelId(selectedCheckpoint);
    const selectionChanged = selectedCheckpoint !== selectedAtStart;
    // Read a TTS-owned slot as empty so the eviction branch clears the stale chat pick.
    const chatActiveModel =
      statusRes.active_model &&
      !isSpeechOnlyStatus(statusRes) &&
      !(statusLoading && options?.externalChatSlotLoad);
    if (
      chatActiveModel &&
      !isExternalSelectionActive &&
      !selectionChanged
    ) {
      const checkpointId = resolveInferenceCheckpointId(statusRes);
      if (checkpointId) {
        const previousGgufVariant =
          useChatRuntimeStore.getState().activeGgufVariant;
        // A model loaded elsewhere replaces the resident; its old pin must not carry over.
        if (
          checkpointId !== selectedCheckpoint ||
          (statusRes.gguf_variant ?? null) !== (previousGgufVariant ?? null)
        ) {
          useChatRuntimeStore.setState({ activeLoadId: null });
        }
        setCheckpoint(checkpointId, statusRes.gguf_variant);
        applyActiveModelStatusToStore(statusRes, {
          previousCheckpoint: selectedCheckpoint,
          previousGgufVariant,
          adoptingExistingServerModel: selectedCheckpoint === "",
        });
        // Re-apply live status: catalog data omits audio capability.
        syncModelCapabilities(checkpointId, statusRes);

        // History can load before this status sets a window, so recount here.
        const hydrated = useChatRuntimeStore.getState();
        if (
          (hydrated.contextUsage == null || hydrated.contextUsage.estimated) &&
          hydrated.activeThreadId != null &&
          hydrated.loadedContextLength != null &&
          !isExternalModelId(checkpointId)
        ) {
          void refreshContextUsage({ threadId: hydrated.activeThreadId });
        }
      }
    } else if (
      !chatActiveModel &&
      !isExternalSelectionActive &&
      !selectionChanged
    ) {
      if (isIdleUnloadedStatus(statusRes, idleUnloadArmed)) return;
      // Image, video or audio loads evict the chat model; clear the stale selection.
      const { residentCheckpoint: wasResident, modelLoading } =
        useChatRuntimeStore.getState();
      // specFallbackReason survives here; clearing activeModelIsLocal alone mislabels the warning.
      useChatRuntimeStore.setState({
        residentCheckpoint: null,
        modelRequiresTrustRemoteCode: false,
        loadedIsMultimodal: false,
        loadedVisionDisabledByUser: null,
        loadedIsDiffusion: false,
      });
      if (
        (wasResident || isSpeechOnlyStatus(statusRes)) &&
        selectedCheckpoint &&
        (!modelLoading || options?.externalChatSlotLoad)
      ) {
        if (wasResident) {
          toast.info(`${wasResident} is no longer loaded`, {
            description:
              "The server released it, which loading an image, video or audio model does. Pick it again to keep chatting.",
          });
        }
        useChatRuntimeStore.getState().clearCheckpoint();
      }
    }
  } catch (error) {
    // A superseded refresh must not toast a stale failure.
    if (signal?.aborted || superseded()) return;
    if (isSilencedDesktopUpdateFailure(error, downWhenIssued)) return;
    const message =
      error instanceof Error ? error.message : "Failed to load models";
    setModelsError(message);
    toast.error("Failed to refresh models", {
      description: message,
    });
  }
}

/** Hydrate from status, then observe until settled. The gate goes up before the sync. */
async function refreshAndWaitForServerModel(options?: {
  signal?: AbortSignal;
  includeLoras?: boolean;
  preserveIdleUnloaded?: boolean;
  externalChatSlotLoad?: boolean;
}): Promise<void> {
  const signal = options?.signal;
  const release = beginServerModelWait(signal);
  try {
    await syncInferenceStatusToStore(options);
    await waitForServerModel(signal);
  } finally {
    release();
  }
}

/** Reconcile after the server unloaded the active local model (e.g. llama.cpp update). */
export async function resyncInferenceStatusAfterServerModelChange(): Promise<void> {
  // A llama.cpp update replaces the binary whose --help the flag catalogue describes.
  invalidateLlamaFlagCatalog();
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

  useEffect(() => {
    let cancelled = false;
    loadMultiModelEnabled("Failed to read the multiple models setting").then(
      (enabled) => {
        if (!cancelled) {
          useChatRuntimeStore.getState().setKeepModelsLoaded(enabled);
        }
      },
      () => undefined,
    );
    return () => {
      cancelled = true;
    };
  }, []);

  const [loadingModel, setLoadingModel] = useState<{
    id: string;
    displayName: string;
    isDownloaded?: boolean;
    isCachedLora?: boolean;
    ggufVariant?: string | null;
    nativePathToken?: string | null;
  } | null>(null);
  const [loadToastDismissed, setLoadToastDismissed] = useState(false);
  const [loadProgress, setLoadProgress] = useState<{
    percent: number | null;
    label: string | null;
    phase: "downloading" | "starting";
  } | null>(null);
  const loadAbortRef = useRef<AbortController | null>(null);
  const loadingModelRef = useRef<typeof loadingModel>(null);
  const loadToastIdRef = useRef<string | number | null>(null);
  const loadToastDismissedRef = useRef(false);
  const cancelUnloadPendingRef = useRef(false);
  const loadLifecycleLeaseRef = useRef<ModelLifecycleLease | null>(null);
  // Async callbacks compare tokens so a superseded run cannot release a newer run's slot.
  const activeLoadRunRef = useRef<ActiveModelLoadRun | null>(null);

  const setLoadToastDismissedState = useCallback((dismissed: boolean) => {
    loadToastDismissedRef.current = dismissed;
    setLoadToastDismissed(dismissed);
  }, []);

  const resetLoadingUi = useCallback(() => {
    const inFlight = loadingModelRef.current;
    setLoadingModel(null);
    setLoadProgress(null);
    loadingModelRef.current = null;
    loadAbortRef.current = null;
    loadToastIdRef.current = null;
    setLoadToastDismissedState(false);
    if (inFlight) {
      useChatRuntimeStore.getState().clearLoadingModelPick(pickOf(inFlight));
    }
    if (!cancelUnloadPendingRef.current) {
      const lease = loadLifecycleLeaseRef.current;
      loadLifecycleLeaseRef.current = null;
      if (lease !== null) {
        useChatRuntimeStore.getState().endModelLoading(lease);
      }
    }
  }, [setLoadToastDismissedState]);

  const resetLoadingUiForRun = useCallback(
    (run: ActiveModelLoadRun) => {
      if (!ownsModelLoadRun(activeLoadRunRef.current, run)) return;
      // Cancellation owns the slot until /unload settles.
      if (run.cancelPromise) return;
      activeLoadRunRef.current = releaseOwnedModelLoadRun(
        activeLoadRunRef.current,
        run,
      );
      resetLoadingUi();
    },
    [resetLoadingUi],
  );

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
    async (options?: {
      waitForServerModel?: boolean;
      signal?: AbortSignal;
      includeLoras?: boolean;
      preserveIdleUnloaded?: boolean;
      externalChatSlotLoad?: boolean;
    }) => {
      if (options?.waitForServerModel) {
        await refreshAndWaitForServerModel(options);
        return;
      }
      await syncInferenceStatusToStore(options);
    },
    [],
  );

  // Nothing polls /status, so re-read whenever another runtime finishes a load.
  useEffect(
    () =>
      subscribeModelLifecycle(({ runtime }) => {
        // STT evicts nothing and chat reconciles itself; TTS takes this very slot.
        if (runtime === "chat" || runtime === "stt") return;
        // Both edges: the arbiter evicts chat inside the load POST, before the download starts.
        void refresh({
          includeLoras: false,
          externalChatSlotLoad: runtime === "tts",
        });
      }),
    [refresh],
  );

  useEffect(
    () =>
      subscribeResidentStatusRefresh(() => {
        void refresh({ includeLoras: false, preserveIdleUnloaded: true });
      }),
    [refresh],
  );

  /** Stop the slot's run and await /unload: the abort signal cannot stop an in-flight POST. */
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

      const model = loadingModelRef.current;
      run.abortController.abort();
      loadAbortRef.current = null;
      loadingModelRef.current = null;
      const tid = loadToastIdRef.current;
      loadToastIdRef.current = null;
      setLoadingModel(null);
      setLoadProgress(null);
      setLoadToastDismissedState(false);
      if (tid != null) toast.dismiss(tid);
      if (model) {
        useChatRuntimeStore.getState().clearLoadingModelPick(pickOf(model));
      }
      const isCachedOrLocal = model?.isDownloaded || model?.isCachedLora;
      toast.info("Stopping model load", {
        description: isCachedOrLocal
          ? undefined
          : "The current download may still finish in the background.",
      });
      const cancelPromise = (async (): Promise<boolean> => {
        try {
          // Unforced: a chat may stream on the previous model. Scoped to this run's cancel id.
          if (run.loadAttemptPath) {
            await unloadModel({
              model_path: run.loadAttemptPath,
              cancel_load_request_id: run.requestId,
            });
          }
          // A standalone cancel can race the backend; derive the checkpoint from status.
          if (!preserveCheckpoint) {
            if (!useChatRuntimeStore.getState().keepModelsLoaded || run.residentModelUnloaded) {
              clearCheckpoint();
            }
            await refresh();
          }
          return true;
        } catch (error) {
          const detail =
            error instanceof Error ? error.message : "Unknown unload error";
          const message = `Failed to stop ${model?.displayName ?? "the model"}`;
          setModelsError(`${message}: ${detail}`);
          toast.error(message, { description: detail });
          // The request failed, so reconcile against the backend before releasing the
          // slot. The caller still receives false and must not start a replacement
          // from an uncertain backend state.
          try {
            // The unabortable /load can still land; read status only once the run has stopped.
            await run.settledPromise;
            await refresh();
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
            // Hold the slot until the aborted coroutine unwinds from its unabortable awaits.
            await run.settledPromise;
            if (run.loadAttemptPath && run.forceCancelActive && !run.residentModelUnloaded) {
              try {
                const status = await getInferenceStatus(undefined, run.rollbackCheckpoint ?? undefined);
                if ((status.loading?.length ?? 0) === 0) {
                  run.residentModelUnloaded = !residentModelMatchesPick(status, {
                    id: run.rollbackCheckpoint ?? "",
                    loadPath: run.rollbackLoadId ?? run.rollbackCheckpoint,
                    ggufVariant: run.rollbackVariant,
                  });
                }
              } catch {
                // An unavailable status is inconclusive; do not clear a checkpoint blindly.
              }
            }
            if (
              (!run.loadAttemptPath && run.residentModelUnloaded) ||
              (run.loadAttemptPath && run.forceCancelActive && run.residentModelUnloaded)
            ) {
              try {
                // The store names the former resident but holds the cancelled target's settings.
                restoreRollbackConfigForClear(run);
                clearCheckpoint();
                await refresh();
              } catch {
                // The cancel's own error reporting stands; the slot is still released.
              }
            }
            activeLoadRunRef.current = releaseOwnedModelLoadRun(
              activeLoadRunRef.current,
              run,
            );
            const lease = loadLifecycleLeaseRef.current;
            loadLifecycleLeaseRef.current = null;
            if (lease !== null) {
              useChatRuntimeStore.getState().endModelLoading(lease);
            }
          }
        }
      })();
      run.cancelPromise = cancelPromise;
      return cancelPromise;
    },
    [clearCheckpoint, refresh, setLoadToastDismissedState, setModelsError],
  );

  const cancelLoadingWithCheckpointPolicy = useCallback(
    (preserveCheckpoint = false): Promise<boolean> => {
      const run = activeLoadRunRef.current;
      return run
        ? cancelLoadRun(run, preserveCheckpoint)
        : Promise.resolve(false);
    },
    [cancelLoadRun],
  );

  const cancelLoading = useCallback(
    (): Promise<boolean> => cancelLoadingWithCheckpointPolicy(false),
    [cancelLoadingWithCheckpointPolicy],
  );
  /** Stop the pending load for a different pick, preserving the checkpoint as rollback target. */
  const cancelLoadingForReplacement = useCallback(
    async (intentId: number): Promise<boolean> => {
      const run = activeLoadRunRef.current;
      pendingExternalReplacement = { intentId, config: run?.rollbackConfig };
      const stopped = await cancelLoadingWithCheckpointPolicy(true);
      // The load may have finished before cancel captured the run; nothing left to stop.
      if (!stopped && !activeLoadRunRef.current) {
        return true;
      }
      if (!stopped && pendingExternalReplacement?.intentId === intentId) {
        pendingExternalReplacement = null;
      }
      return stopped;
    },
    [cancelLoadingWithCheckpointPolicy],
  );

  const invalidatePendingModelSelection = useCallback((): number => {
    modelSelectionIntentEpoch += 1;
    return modelSelectionIntentEpoch;
  }, []);

  const discardExternalReplacement = useCallback((intentId: number): void => {
    if (pendingExternalReplacement?.intentId === intentId) {
      pendingExternalReplacement = null;
      pendingReplacementRollback = null;
    }
  }, []);

  const restoreConfigForExternalReplacement = useCallback(
    (intentId: number): void => {
      const config =
        (pendingExternalReplacement?.intentId === intentId
          ? pendingExternalReplacement.config
          : undefined) ?? pendingReplacementRollback?.config;
      if (config) {
        applyPerModelConfigToRuntime(config, {
          isDiffusion: useChatRuntimeStore.getState().loadedIsDiffusion,
        });
      }
      discardExternalReplacement(intentId);
      pendingReplacementRollback = null;
    },
    [discardExternalReplacement],
  );

  const isModelSelectionIntentCurrent = useCallback(
    (intentId: number) => modelSelectionIntentEpoch === intentId,
    [],
  );

  const selectModel = useCallback(
    async (selection: string | SelectedModelInput) => {
      const modelId = typeof selection === "string" ? selection : selection.id;
      const loadPath =
        (typeof selection === "string" ? null : selection.loadId) || modelId;
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
      let isDiffusion =
        typeof selection === "string" ? undefined : selection.isDiffusion;
      let previousConfigForReplacement =
        typeof selection === "string" ? undefined : selection.previousConfig;
      const restorePreviousConfig = () => {
        if (previousConfigForReplacement) {
          applyPerModelConfigToRuntime(previousConfigForReplacement, {
            isDiffusion:
              useChatRuntimeStore.getState().loadedIsDiffusion,
          });
        }
      };
      const throwOnError =
        typeof selection === "string" ? false : selection.throwOnError ?? false;
      const keepSpeculative =
        typeof selection === "string" ? false : selection.keepSpeculative ?? false;
      const currentVariant = useChatRuntimeStore.getState().activeGgufVariant;
      const initiallyLoadingSamePick =
        loadingModelRef.current?.id === modelId &&
        (loadingModelRef.current?.ggufVariant ?? null) === (ggufVariant ?? null) &&
        (loadingModelRef.current?.nativePathToken ?? null) ===
          (nativePathToken ?? null);
      if (
        !forceReload &&
        (!modelId ||
          (params.checkpoint === modelId &&
            (ggufVariant ?? null) === (currentVariant ?? null) &&
            !loadingModelRef.current &&
            !useChatRuntimeStore.getState().loadingModelPick &&
            !pendingReplacementRollback))
      ) {
        if (modelId) modelSelectionIntentEpoch += 1;
        restorePreviousConfig();
        return;
      }
      if (
        initiallyLoadingSamePick &&
        !activeLoadRunRef.current?.cancelPromise
      ) {
        restorePreviousConfig();
        return;
      }
      dismissStartToastsForModelSelection();

      // Register intent before awaiting cancellation so stale cleanup cannot clobber new config.
      const loadIntentId = ++modelSelectionIntentEpoch;
      const activeRunBeforeCredentials = activeLoadRunRef.current;

      const isLocal = isLocalModelPath(modelId);
      let hfToken = useChatRuntimeStore.getState().hfToken || null;
      // Credential prompts complete before cancelling, so a decline does not strand the resident.
      const mayReachHub =
        !isLocal && !isOllamaModelId(modelId) && nativePathToken == null;
      if (mayReachHub) {
        const preparedToken = await prepareHfTokenForUse(hfToken);
        if (!preparedToken.proceed) {
          if (modelSelectionIntentEpoch === loadIntentId) {
            toast.error("Model load cancelled.");
            // The prior run may fail while credentials are open; wait for it, then reconcile.
            if (activeRunBeforeCredentials) {
              void activeRunBeforeCredentials.settledPromise.then(async () => {
                if (modelSelectionIntentEpoch !== loadIntentId) return;
                if (!activeRunBeforeCredentials.residentModelUnloaded) {
                  const current = useChatRuntimeStore.getState();
                  if (
                    current.params.checkpoint ===
                    activeRunBeforeCredentials.rollbackCheckpoint
                  ) {
                    restoreRollbackConfigForClear(activeRunBeforeCredentials);
                  }
                  return;
                }
                if (activeRunBeforeCredentials.residentModelUnloaded) {
                  try {
                    const status = await getInferenceStatus(
                      undefined,
                      activeRunBeforeCredentials.rollbackCheckpoint ?? undefined,
                    );
                    if (modelSelectionIntentEpoch !== loadIntentId) return;
                    if (
                      (status.loading?.length ?? 0) === 0 &&
                      useChatRuntimeStore.getState().params.checkpoint ===
                        activeRunBeforeCredentials.rollbackCheckpoint
                    ) {
                      if (!status.active_model) {
                        restoreRollbackConfigForClear(activeRunBeforeCredentials);
                        useChatRuntimeStore.getState().clearCheckpoint();
                      } else if (
                        residentModelMatchesPick(status, {
                          id: activeRunBeforeCredentials.rollbackCheckpoint ?? "",
                          loadPath:
                            activeRunBeforeCredentials.rollbackLoadId ??
                            activeRunBeforeCredentials.rollbackCheckpoint,
                          ggufVariant: activeRunBeforeCredentials.rollbackVariant,
                        })
                      ) {
                        restoreRollbackConfigForClear(activeRunBeforeCredentials);
                      }
                      await refresh();
                    }
                  } catch {
                    // Preserve state when backend status is unavailable; a later refresh can reconcile it.
                  }
                  return;
                }
              });
            }
          }
          return;
        }
        hfToken = preparedToken.token;
      }
      if (modelSelectionIntentEpoch !== loadIntentId) return;
      // A prior run may have failed while this pick was waiting for Hub credentials. If it
      // never unloaded the resident, it could not restore its staged config after losing intent.
      if (activeRunBeforeCredentials && activeLoadRunRef.current !== activeRunBeforeCredentials) {
        const current = useChatRuntimeStore.getState();
        if (
          !activeRunBeforeCredentials.residentModelUnloaded &&
          !activeRunBeforeCredentials.loadAttemptPath &&
          current.params.checkpoint === activeRunBeforeCredentials.rollbackCheckpoint
        ) {
          if (activeRunBeforeCredentials.rollbackConfig) {
            previousConfigForReplacement = activeRunBeforeCredentials.rollbackConfig;
            restoreRollbackConfigForClear(activeRunBeforeCredentials);
          }
        } else if (
          activeRunBeforeCredentials.loadAttemptPath &&
          current.params.checkpoint != null
        ) {
          if (
            current.params.checkpoint !== activeRunBeforeCredentials.rollbackCheckpoint
          ) {
            previousConfigForReplacement = currentRuntimePerModelConfig({
              includeMaxSeqLength: true,
            });
          } else if (activeRunBeforeCredentials.residentModelUnloaded) {
            // A failed compensation reload leaves the checkpoint in the store but nothing resident.
            try {
              const status = await getInferenceStatus(
                undefined,
                activeRunBeforeCredentials.rollbackCheckpoint ?? undefined,
              );
              if (
                (status.loading?.length ?? 0) === 0 &&
                residentModelMatchesPick(status, {
                  id: activeRunBeforeCredentials.rollbackCheckpoint ?? "",
                  loadPath:
                    activeRunBeforeCredentials.rollbackLoadId ??
                    activeRunBeforeCredentials.rollbackCheckpoint,
                  ggufVariant: activeRunBeforeCredentials.rollbackVariant,
                })
              ) {
                previousConfigForReplacement = currentRuntimePerModelConfig({
                  includeMaxSeqLength: true,
                });
              }
            } catch {
              // Keep the existing snapshot if backend status is unavailable.
            }
          }
        }
      }




      if (pendingReplacementRollback?.config) {
        previousConfigForReplacement = pendingReplacementRollback.config;
      }
      // The cancelled run's own rollback target is the model that was working before
      // it, not the transient state it may already have applied. Adopt that target,
      // or a failed replacement restores the wrong model's runtime settings.
      const inheritCancelledRunRollback = (cancelledRun: ActiveModelLoadRun) => {
        if (cancelledRun.rollbackConfig) {
          previousConfigForReplacement = cancelledRun.rollbackConfig;
        }
        pendingReplacementRollback = {
          checkpoint: cancelledRun.rollbackCheckpoint,
          config: previousConfigForReplacement,
          // Variant and load id travel along: clearCheckpoint() drops them from the store.
          residentUnloaded: cancelledRun.residentModelUnloaded,
          variant: cancelledRun.rollbackVariant ?? null,
          loadId: cancelledRun.rollbackLoadId,
          nativePathToken: cancelledRun.rollbackNativePathToken,
          nativePathExpiresAtMs: cancelledRun.rollbackNativePathExpiresAtMs,
          loadedState: cancelledRun.rollbackLoadedState,
        };
      };

      // A different pick supersedes the load in flight. Await its cancellation and the
      // backend unload before claiming the slot, then re-check: the load can settle
      // between the two, and another selection may be waiting on the same run.
      while (true) {
        const activeRun = activeLoadRunRef.current;
        const inFlightLoad =
          loadingModelRef.current ??
          useChatRuntimeStore.getState().loadingModelPick;
        // Break only when neither picker entry nor run remains, or a mid-cancel pick is dropped.
        if (!inFlightLoad && !activeRun) break;
        if (inFlightLoad) {
          const loadingSamePick =
            inFlightLoad.id === modelId &&
            (inFlightLoad.ggufVariant ?? null) === (ggufVariant ?? null) &&
            (inFlightLoad.nativePathToken ?? null) === (nativePathToken ?? null);
          if (loadingSamePick && !activeRun?.cancelPromise) {
            restorePreviousConfig();
            return;
          }
          if (!activeRun) {
            // Published but unowned: the other caller is still in preflight.
            restorePreviousConfig();
            return;
          }
        }
        if (!activeRun) break;
        inheritCancelledRunRollback(activeRun);
        const stopped = await cancelLoadRun(activeRun, true);
        // Cancellation can settle after release, so adopt the target again.
        inheritCancelledRunRollback(activeRun);
        if (modelSelectionIntentEpoch !== loadIntentId) {
          if (throwOnError) {
            throw new Error("Model selection was superseded by a newer choice.");
          }
          return;
        }
        if (!stopped) {
          restorePreviousConfig();
          // The unload failed, so the backend state is uncertain; drop the inherited rollback.
          pendingReplacementRollback = null;
          const message =
            "The current model could not be stopped, so the new model was not loaded.";
          setModelsError(message);
          if (throwOnError) throw new Error(message);
          return;
        }
      }

      // A local pick that is superseded by a later selection must not keep the slot.
      // The replacement loop only judges the load present when it ran. A rival pick can
      // still claim the slot during the awaits below, and it owns the resident model
      // now, so the adopt and confirm paths must yield to it rather than adopt over it.
      const rivalLoadStarted = (): boolean => {
        const rival = useChatRuntimeStore.getState().loadingModelPick;
        if (!rival) return false;
        return !(
          rival.id === modelId &&
          (rival.ggufVariant ?? null) === (ggufVariant ?? null) &&
          (rival.nativePathToken ?? null) === (nativePathToken ?? null)
        );
      };
      if (modelSelectionIntentEpoch !== loadIntentId) {
        if (throwOnError) {
          throw new Error("Model selection was superseded by a newer choice.");
        }
        return;
      }
      // Ask the backend, not params.checkpoint: an external pick leaves the local model resident.
      const selectedCheckpoint =
        useChatRuntimeStore.getState().params.checkpoint;
      const pendingConfig =
        typeof selection !== "string" ? selection.config : undefined;
      if (!forceReload && !nativePathToken) {
        const readPickStatus = () =>
          getInferenceStatus(undefined, modelId).catch(() => null);
        const residentStatus = await readPickStatus();
        // Warm the GPU cache first: a cold cache passes the pick through unvalidated.
        if (residentStatus && pendingConfig?.selectedGpuIds !== undefined) {
          await ensureGpuDeviceCache().catch(() => {});
        }
        // Mirror what /load would send for an unconfigured pick; performLoad clears per-model fields.
        const live = useChatRuntimeStore.getState();
        const resetsPerModelSettings = Boolean(
          live.params.checkpoint &&
            (live.params.checkpoint !== modelId ||
              (live.activeGgufVariant ?? null) !== (ggufVariant ?? null)) &&
            !keepSpeculative,
        );
        const comparedConfig =
          pendingConfig ??
          (resetsPerModelSettings
            ? {
                ...DEFAULT_PER_MODEL_CONFIG,
                kvCacheDtype: live.kvCacheDtype ?? null,
                tensorParallel: live.tensorParallel ?? false,
                gpuMemoryMode: live.gpuMemoryMode,
              }
            : currentRuntimePerModelConfig());
        // 0 is the catalogue's "unknown", which the comparison reads as a reload.
        const managedFlags = residentStatus
          ? await loadManagedLlamaFlags().catch(() => null)
          : null;
        // Hoisted to run twice: another tab can swap the resident model during the awaits.
        const adoptable = (status: InferenceStatusResponse) =>
          (status.loading?.length ?? 0) === 0 &&
          (status.engine ?? "auto") === (comparedConfig.engine ?? "auto") &&
          residentModelMatchesPick(status, {
            id: modelId,
            loadPath,
            ggufVariant,
          }) &&
          // Do not skip /load while the audio probe is pending, or audio stays undetected.
          status.audio_probe_pending !== true &&
          // A repairable speculative fallback must reload to retry the drafter.
          !residentSpeculativeNeedsRepair(
            status,
            normalizeSpeculativeType(pendingConfig?.speculativeType) ??
              readPersistedSpeculativeType(),
            (loadPath ?? modelId).toLowerCase().endsWith(".gguf"),
          ) &&
          // Identity alone is not enough: differing launch config is a real reload.
          residentRuntimeMatchesConfig(status, comparedConfig, {
            speculativeType: readPersistedSpeculativeType(),
            gpuMemoryMode: readPersistedGpuMemoryMode(),
            gpuLayers: GPU_LAYERS_AUTO,
            nCpuMoe: 0,
            resolveContextLength: (customContextLength) => {
              const live = useChatRuntimeStore.getState();
              const platform = usePlatformStore.getState();
              const residentIsGguf = status.is_gguf ?? false;
              return resolveLoadMaxSeqLength({
                modelId,
                ggufVariant,
                isGguf: residentIsGguf,
                customContextLength,
                loadedContextLength: live.loadedContextLength,
                currentCheckpoint: live.params.checkpoint,
                activeGgufVariant: live.activeGgufVariant,
                isMlx: isServedByMlx(
                  residentIsGguf,
                  platform.deviceType,
                  platform.chatOnlyReason,
                ),
                pinnedMaxSeqLength: normalizeMaxSeqLength(
                  pendingConfig
                    ? pendingConfig.maxSeqLength
                    : resolveInitialConfig(modelId, ggufVariant).config
                        .maxSeqLength,
                ),
                defaultMaxSeqLength: live.params.maxSeqLength || DEFAULT_MAX_SEQ_LENGTH,
                presetSource: live.activePresetSource,
              });
            },
            parallelSlots: managedFlags?.defaultParallelSlots || null,
            splitRatio:
              pendingConfig?.tensorSplit !== undefined
                ? pendingConfig.tensorSplit
                : resetsPerModelSettings
                  ? null
                  : useChatRuntimeStore.getState().splitRatio,
            reconcileGpuIds: (ids, savedIndexKind) =>
              reconcilePersistedGpuIds(
                ids,
                savedIndexKind,
                status.is_diffusion ?? false,
              ),
            defaultEngineGpuIds: defaultEngineGpuIds(),
            normalizeSpeculative: normalizeSpeculativeType,
          });
        if (
          residentStatus &&
          adoptable(residentStatus) &&
          // Server-wide settings: a save must still force a reload even if the pick is identical.
          !(await readServerWideReloadHints())
        ) {
          // Re-read status: the earlier one may describe a model swapped out during the awaits.
          const confirmedStatus = await readPickStatus();
          if (confirmedStatus && adoptable(confirmedStatus)) {
            if (rivalLoadStarted()) return;
            if (modelSelectionIntentEpoch !== loadIntentId) return;
            // Roll back the pre-applied config before hydrating so the resident status wins.
            restorePreviousConfig();
            // maxSeqLength is client-side and no status echoes it, so keep this pick's cap.
            const pickedMaxSeqLength =
              normalizeMaxSeqLength(pendingConfig?.maxSeqLength) ??
              defaultInferenceParams.maxSeqLength;
            const restored = useChatRuntimeStore.getState();
            if (restored.params.maxSeqLength !== pickedMaxSeqLength) {
              restored.setParams({
                ...restored.params,
                maxSeqLength: pickedMaxSeqLength,
              });
            }
            const previousGgufVariant =
              useChatRuntimeStore.getState().activeGgufVariant;
            // Adopt this pick's own pin, or Apply would reload an earlier resident.
            useChatRuntimeStore.setState({
              activeLoadId: loadPath === modelId ? null : loadPath,
            });
            useChatRuntimeStore
              .getState()
              .setCheckpoint(modelId, confirmedStatus.gguf_variant);
            applyActiveModelStatusToStore(confirmedStatus, {
              previousCheckpoint: selectedCheckpoint,
              previousGgufVariant,
            });
            syncModelCapabilities(modelId, confirmedStatus);
            // Keep the pick's GPU selection; hydration would widen it back.
            if (pendingConfig?.selectedGpuIds !== undefined) {
              const picked = reconcilePersistedGpuIds(
                pendingConfig.selectedGpuIds,
                pendingConfig.selectedGpuIndexKind,
                confirmedStatus.is_diffusion ?? false,
              );
              const hydrated = useChatRuntimeStore.getState();
              if (!sameGpuSelection(hydrated.selectedGpuIds, picked)) {
                useChatRuntimeStore.setState({
                  selectedGpuIds: picked,
                  loadedGpuIds: picked,
                });
              }
            }
            // Adopting completes the replacement without a load run; consume the inherited rollback.
            pendingReplacementRollback = null;

            void refreshContextUsage({ afterModelLoad: true });
            return;
          }
        }
      }

      // Hold the lifecycle lease through confirmation and loading. A PREFLIGHT holder (parked on
      // the stop-running-chats confirmation) owns the lease without having published a run or a
      // picker entry, so a pick arriving now reads null from the gate while the holder later
      // yields the slot as stale -- and both picks are lost. Wait, bounded, for the holder to
      // release the lease, then claim it: the user's latest pick must not be dropped by a lease
      // whose owner is still deciding.
      let lifecycleLease: ModelLifecycleLease | null = null;
      while (lifecycleLease === null) {
        lifecycleLease = useChatRuntimeStore
          .getState()
          .beginModelLoading("preparing");
        if (lifecycleLease !== null) break;
        if (modelSelectionIntentEpoch !== loadIntentId) return;
        await new Promise<void>((resolve) => {
          let settled = false;
          const unsubscribe = useChatRuntimeStore.subscribe((state) => {
            if (settled || state.modelLoading) return;
            settled = true;
            unsubscribe();
            resolve();
          });
          setTimeout(() => {
            if (settled) return;
            settled = true;
            unsubscribe();
            resolve();
          }, PREFLIGHT_LEASE_RETRY_MS);
        });
        if (modelSelectionIntentEpoch !== loadIntentId) return;
      }
      if (lifecycleLease === null) {
        restorePreviousConfig();
        toast.info("A model is loading", {
          description: "Wait for it to finish or cancel it first.",
        });
        return;
      }
      loadLifecycleLeaseRef.current = lifecycleLease;
      const releasePreflightLifecycleLease = () => {
        if (loadLifecycleLeaseRef.current !== lifecycleLease) {
          return;
        }
        loadLifecycleLeaseRef.current = null;
        useChatRuntimeStore.getState().endModelLoading(lifecycleLease);
      };

      // Every chat decodes on the server this load replaces, so confirm before cancelling.
      let stopDecision: Awaited<
        ReturnType<typeof confirmStopRunningChatsIfNeeded>
      >;
      const { keepModelsLoaded, loadedModels: loadedNow, params: paramsNow } =
        useChatRuntimeStore.getState();
      const keepsOthers =
        keepModelsLoaded && !forceReload && (paramsNow.engine ?? "auto") === "auto";
      const switchingNote = keepsOthers ? "Keeping the loaded models." : "Switching models.";
      const touchesOnlySelected =
        forceReload && !isExternalModelId(paramsNow.checkpoint) && loadedNow.length > 1;
      try {
        stopDecision = keepsOthers
          ? {
              proceed: true,
              forceCancelActive: false,
              promptQueueThreadIds: [],
              preStreamRunTokens: [],
            }
          : await confirmStopRunningChatsIfNeeded(
              forceReload ? "Applying these settings" : "Loading a different model",
              "reload",
              touchesOnlySelected ? (paramsNow.checkpoint ?? undefined) : undefined,
            );
      } catch (error) {
        releasePreflightLifecycleLease();
        throw error;
      }
      if (!stopDecision.proceed) {
        releasePreflightLifecycleLease();
        // Restore the inherited target, not this pick's previousConfig (may be a superseded config).
        restorePreviousConfig();
        // After a decline, drop the rollback unless the resident was already unloaded or reowned.
        if (
          modelSelectionIntentEpoch === loadIntentId &&
          pendingReplacementRollback?.residentUnloaded === false
        ) {
          pendingReplacementRollback = null;
        }
        return;
      }
      // Re-check the tracked picker for a load that was already starting when this lifecycle lease was acquired.
      try {
        // A newer pick supersedes this preflight even without the lease.
        if (rivalLoadStarted() || modelSelectionIntentEpoch !== loadIntentId) {
          releasePreflightLifecycleLease();
          return;
        }
      } catch (error) {
        releasePreflightLifecycleLease();
        throw error;
      }
      const forceCancelActive = stopDecision.forceCancelActive;

      const explicitIsLora =
        typeof selection === "string" ? undefined : selection.isLora;
      const extraLoadingDescription =
        typeof selection === "string" ? undefined : selection.loadingDescription;
      const isDownloaded =
        typeof selection === "string" ? false : selection.isDownloaded ?? false;
      const model = models.find((entry) => entry.id === modelId);
      const lora = loras.find((entry) => entry.id === modelId);
      // A native path token is a local GGUF even if its display id lacks ".gguf".
      const isGguf =
        explicitIsGguf ??
        (ggufVariant != null ||
          nativePathToken != null ||
          model?.isGguf === true);
      const loraIsAdapter = lora?.exportType === "lora";
      let isLora =
        explicitIsLora ?? model?.isLora ?? loraIsAdapter ?? false;
      const displayName = model?.name || lora?.name || modelId;
      const toastDisplayName = shortModelLabel(displayName);
      const currentRollbackState = useChatRuntimeStore.getState();
      const inheritedPendingRollback = pendingReplacementRollback;
      if (inheritedPendingRollback?.config) {
        previousConfigForReplacement = inheritedPendingRollback.config;
      }
      const currentCheckpoint = currentRollbackState.params.checkpoint;
      const previousCheckpoint = inheritedPendingRollback
        ? inheritedPendingRollback.checkpoint
        : currentCheckpoint;
      const previousVariant = inheritedPendingRollback
        ? (inheritedPendingRollback.variant ?? null)
        : (currentRollbackState.activeGgufVariant ?? null);
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
      const isCachedLora = isLora && isLocal;
      let loadingDescription = [
        currentCheckpoint ? switchingNote : null,
        extraLoadingDescription ?? null,
        isDownloaded ? "Loading cached model into memory." : null,
        !isDownloaded && isCachedLora ? "Loading trained model into memory." : null,
      ]
        .filter(Boolean)
        .join(" ");
      setModelsError(null);
      setLastModelLoadError(null);
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
      useChatRuntimeStore.getState().setLoadingModelPick(pickOf(loadInfo));
      setLoadProgress(
        isDownloaded || isCachedLora
          ? { percent: null, label: null, phase: "starting" }
          : { percent: 0, label: "Preparing download", phase: "downloading" },
      );
      loadingModelRef.current = loadInfo;
      const abortCtrl = new AbortController();
      loadAbortRef.current = abortCtrl;
      // The resolver is assigned synchronously so a cancel can await it mid-preflight.
      let markLoadRunSettled = () => {};
      const loadRunSettled = new Promise<void>((resolve) => {
        markLoadRunSettled = resolve;
      });
      const loadRun: ActiveModelLoadRun = {
        attemptId: loadIntentId,
        intentId: loadIntentId,
        abortController: abortCtrl,
        cancelPromise: null,
        rollbackCheckpoint: previousCheckpoint,
        rollbackConfig: previousConfigForReplacement,
        // Carried on the run: clearCheckpoint() drops the store's activeGgufVariant.
        rollbackVariant: previousVariant,
        rollbackLoadId: inheritedPendingRollback
          ? inheritedPendingRollback.loadId ?? null
          : currentRollbackState.activeLoadId ?? null,
        rollbackNativePathToken: inheritedPendingRollback
          ? inheritedPendingRollback.nativePathToken ?? null
          : currentRollbackState.activeNativePathToken ?? null,
        rollbackNativePathExpiresAtMs: inheritedPendingRollback
          ? inheritedPendingRollback.nativePathExpiresAtMs ?? null
          : currentRollbackState.activeNativePathExpiresAtMs ?? null,
        rollbackLoadedState: inheritedPendingRollback?.loadedState ?? currentRollbackState,
        residentModelUnloaded: inheritedPendingRollback?.residentUnloaded === true,
        forceCancelActive,
        settledPromise: loadRunSettled,
        markSettled: markLoadRunSettled,

        requestId: crypto.randomUUID(),
        loadAttemptPath: null,
      };
      activeLoadRunRef.current = loadRun;
      pendingReplacementRollback = null;
      const postLoadRefresh = { needed: false };
      let progressModelIds = [modelId];
      let mlxLoadProgress = false;
      const liveBeforeLoad = useChatRuntimeStore.getState();
      if (
        (typeof selection === "string" || !selection.config) &&
        !keepSpeculative &&
        liveBeforeLoad.params.checkpoint &&
        (liveBeforeLoad.params.checkpoint !== modelId ||
          (liveBeforeLoad.activeGgufVariant ?? null) !== (ggufVariant ?? null)) &&
        (liveBeforeLoad.params.engine ?? "auto") !== "auto"
      ) {
        liveBeforeLoad.setParams({
          ...liveBeforeLoad.params,
          engine: "auto",
          enginePrecision: "auto",
          engineParallelism: "tensor",
        });
      }
      const requestedEngine =
        (typeof selection !== "string" ? selection.config?.engine : undefined) ??
        useChatRuntimeStore.getState().params.engine ?? "auto";
      const managedLoad = !isGguf && requestedEngine !== "auto";
      let downloadComplete = isDownloaded || isCachedLora || managedLoad;
      let cpuFallbackReason: CpuFallbackReason | null = null;
      let mmprojFallbackReason: MmprojFallbackReason | null = null;
      let offloadCounts: OffloadCounts = {};
      try {
        async function performLoad(): Promise<void> {
          if (abortCtrl.signal.aborted) throw new Error("Cancelled");
          // The cancelled run may have unloaded the rollback model; ensure the compensating reload runs.
          let previousWasUnloaded =
            inheritedPendingRollback?.residentUnloaded === true;
          const pendingLoadConfig =
            typeof selection !== "string" ? selection.config : undefined;
          // The outgoing model's slot INTENT (blank = follow the server default), which the resolved baseline
          // cannot express. previousConfig is the snapshot taken before pre-applying the target's config,
          // read before the staged-metadata await so a change cannot perturb it. Unlike tensorParallel,
          // vision does NOT survive a model switch: it is per-model config defaulting to vision on, so a
          // target that saved none gets that default. The dedupe above builds its own view of an
          // Every rollback read below uses the INHERITED target, never this pick's own
          // `selection.previousConfig`: when this pick superseded a load that had already
          // pre-applied its settings, that field holds the superseded target's transient config,
          // and a rollback would restore the resident model wearing the wrong model's settings.
          // `previousConfigForReplacement` already carries the resident model's own config.
          const rollbackConfig = previousConfigForReplacement;
          const previousNParallel =
            rollbackConfig
              ? (rollbackConfig.nParallel ?? null)
              : useChatRuntimeStore.getState().nParallel;
          const previousReasoningBudget =
            rollbackConfig
              ? rollbackConfig.reasoningBudget
              : useChatRuntimeStore.getState().reasoningBudget;
          const previousReasoningBudgetMessage =
            rollbackConfig
              ? rollbackConfig.reasoningBudgetMessage
              : useChatRuntimeStore.getState().reasoningBudgetMessage;
          const previousNBatch =
            rollbackConfig
              ? (rollbackConfig.nBatch ?? null)
              : useChatRuntimeStore.getState().nBatch;
          const previousNUbatch =
            rollbackConfig
              ? (rollbackConfig.nUbatch ?? null)
              : useChatRuntimeStore.getState().nUbatch;
          const previousServerTuning: ServerTuningValues =
            rollbackConfig ?? useChatRuntimeStore.getState();
          const previousMlxKvQuant = rollbackConfig
            ? (rollbackConfig.mlxKvQuant ?? null)
            : useChatRuntimeStore.getState().mlxKvQuant;
          const previousMlxInt8Prefill = rollbackConfig
            ? (rollbackConfig.mlxInt8Prefill ?? false)
            : useChatRuntimeStore.getState().mlxInt8Prefill;
          if (isGguf && isDiffusion === undefined) {
            // Prepare the token like validateModel: the Hub 401s an invalid header even for public repos.
            isDiffusion = (
              await fetchGgufStagedMetadata({
                model_path: loadPath,
                gguf_variant: ggufVariant ?? null,
                hf_token: hfToken,
                nativePathToken: nativePathToken ?? null,
              })
            ).isDiffusion;
          }
          // The staged-metadata read cannot be aborted, so check after it.
          if (abortCtrl.signal.aborted) throw new Error("Cancelled");
          const targetIsDiffusion = isDiffusion === true;
          if (pendingLoadConfig) {
            applyPerModelConfigToRuntime(pendingLoadConfig, {
              isDiffusion: targetIsDiffusion,
            });
          }
          const currentCheckpoint =
            useChatRuntimeStore.getState().params.checkpoint;
          const stateBeforeUnload = useChatRuntimeStore.getState();
          const rollbackState = inheritedPendingRollback?.loadedState ?? stateBeforeUnload;
          const platform = usePlatformStore.getState();
          let trustRemoteCode = stateBeforeUnload.params.trustRemoteCode ?? false;
          let approvedRemoteCodeFingerprint: string | null = null;
          const pinnedMaxSeqLength = normalizeMaxSeqLength(
            pendingLoadConfig
              ? pendingLoadConfig.maxSeqLength
              : resolveInitialConfig(modelId, ggufVariant).config.maxSeqLength,
          );
          const maxSeqLength =
            pinnedMaxSeqLength ?? stateBeforeUnload.params.maxSeqLength;
          const previousActiveNativePathToken = inheritedPendingRollback
            ? inheritedPendingRollback.nativePathToken ?? null
            : stateBeforeUnload.activeNativePathToken;
          const previousActiveLoadId = inheritedPendingRollback
            ? inheritedPendingRollback.loadId ?? null
            : stateBeforeUnload.activeLoadId;
          const previousActiveNativePathExpiresAtMs = inheritedPendingRollback
            ? inheritedPendingRollback.nativePathExpiresAtMs ?? null
            : stateBeforeUnload.activeNativePathExpiresAtMs;
          const previousIsGguf =
            previousModel?.isGguf === true
            || previousVariant != null
            || previousActiveNativePathToken != null
            || (previousCheckpoint?.toLowerCase().endsWith(".gguf") ?? false);
          // previousConfig predates pre-applying the next model, so it holds the old context.
          const previousMaxSeqLength =
            previousConfigForReplacement?.maxSeqLength ?? maxSeqLength;
          const previousIsMlx = residentIsServedByMlx(
            previousIsGguf,
            platform.deviceType,
            platform.chatOnlyReason,
            rollbackState.loadedIsMlx,
          );
          // What the outgoing model loaded with, not an unapplied control value.
          const previousPin = rollbackState.loadedCustomContextLength;
          const rollbackMaxSeqLength = previousIsGguf
            ? resolveFitMaxSeqLength(
                previousIsGguf,
                rollbackState.loadedGpuMemoryMode ?? "auto",
                rollbackState.loadedGpuLayers ?? GPU_LAYERS_AUTO,
                rollbackState.loadedCustomContextLength,
                rollbackState.loadedContextLength ?? 0,
              )
            : (previousPin ??
              unpinnedLoadContext(false, previousIsMlx, previousMaxSeqLength));
          // Do not re-read the raw stored token here: it would undo the credential preparation.
          const previousModelRequiresTrustRemoteCode =
            stateBeforeUnload.modelRequiresTrustRemoteCode;
          // Snapshot at click time: React may not have flushed NumericValueInput's blur commit.
          let loadChatTemplateOverride =
            pendingLoadConfig?.chatTemplateOverride?.trim()
              ? pendingLoadConfig.chatTemplateOverride
              : stateBeforeUnload.chatTemplateOverride;
          const loadKvCacheDtype =
            pendingLoadConfig?.kvCacheDtype ?? stateBeforeUnload.kvCacheDtype;
          let loadMlxKvQuant =
            pendingLoadConfig
              ? pendingLoadConfig.mlxKvQuant ?? null
              : stateBeforeUnload.mlxKvQuant;
          let loadMlxInt8Prefill = pendingLoadConfig
            ? (pendingLoadConfig.mlxInt8Prefill ?? false)
            : stateBeforeUnload.mlxInt8Prefill;
          // gpuMemoryMode is standing; per-model knobs are re-baselined with the reset below.
          let loadCustomContextLength =
            pendingLoadConfig?.customContextLength ??
            stateBeforeUnload.customContextLength;
          const loadContextLength = stateBeforeUnload.loadedContextLength;
          const loadTensorParallel = targetIsDiffusion
            ? false
            : (pendingLoadConfig?.tensorParallel ??
              stateBeforeUnload.tensorParallel);
          // Vision does not survive a model switch: it is per-model config defaulting to on.
          const loadSwitchesModelOrVariant = Boolean(
            currentCheckpoint &&
              (currentCheckpoint !== modelId ||
                (stateBeforeUnload.activeGgufVariant ?? null) !==
                  (ggufVariant ?? null)) &&
              !keepSpeculative,
          );
          const loadDisableVision = targetIsDiffusion
            ? false
            : (pendingLoadConfig?.disableVision ??
              (loadSwitchesModelOrVariant
                ? DEFAULT_PER_MODEL_CONFIG.disableVision
                : stateBeforeUnload.disableVision));
          const loadActivePresetSource = stateBeforeUnload.activePresetSource;
          const loadActiveGgufVariant = stateBeforeUnload.activeGgufVariant;
          const loadGpuMemoryMode =
            pendingLoadConfig?.gpuMemoryMode ?? stateBeforeUnload.gpuMemoryMode;
          let loadGpuLayers =
            pendingLoadConfig?.gpuLayers ?? stateBeforeUnload.gpuLayers;
          let loadNCpuMoe =
            pendingLoadConfig?.nCpuMoe ?? stateBeforeUnload.nCpuMoe;
          let loadSplitRatio =
            pendingLoadConfig?.tensorSplit !== undefined
              ? pendingLoadConfig.tensorSplit
              : stateBeforeUnload.splitRatio;
          // Drop stale cross-host GPU picks before /load; warm the device cache first.
          if (
            pendingLoadConfig?.selectedGpuIds !== undefined ||
            stateBeforeUnload.selectedGpuIds != null
          ) {
            await ensureGpuDeviceCache();
          }
          const stagedGpuIds =
            pendingLoadConfig?.selectedGpuIds !== undefined
              ? reconcilePersistedGpuIds(
                  pendingLoadConfig.selectedGpuIds,
                  pendingLoadConfig.selectedGpuIndexKind,
                  targetIsDiffusion,
                )
              : null;
          let loadSelectedGpuIds =
            pendingLoadConfig?.selectedGpuIds !== undefined
              ? stagedGpuIds
              : reconcilePersistedGpuIds(
                  stateBeforeUnload.selectedGpuIds,
                  stateBeforeUnload.selectedGpuIndexKind,
                  targetIsDiffusion,
                );
          let loadSpeculativeType =
            pendingLoadConfig?.speculativeType != null
              ? normalizeSpeculativeType(pendingLoadConfig.speculativeType)
              : stateBeforeUnload.speculativeType;
          let loadSpecDraftNMax =
            pendingLoadConfig?.specDraftNMax ?? stateBeforeUnload.specDraftNMax;
          let loadNParallel =
            pendingLoadConfig?.nParallel ?? stateBeforeUnload.nParallel;
          let loadReasoningBudget =
            pendingLoadConfig?.reasoningBudget ??
            stateBeforeUnload.reasoningBudget;
          let loadReasoningBudgetMessage =
            pendingLoadConfig?.reasoningBudgetMessage ??
            stateBeforeUnload.reasoningBudgetMessage;
          // No fallback: undefined means unread, and the route keeps stored flags when omitted.
          const loadLlamaExtraArgs = pendingLoadConfig?.llamaExtraArgs;
          let loadNBatch =
            pendingLoadConfig?.nBatch ?? stateBeforeUnload.nBatch;
          let loadNUbatch =
            pendingLoadConfig?.nUbatch ?? stateBeforeUnload.nUbatch;
          let loadServerTuning: ServerTuningValues = {
            loadMode: pendingLoadConfig?.loadMode ?? stateBeforeUnload.loadMode,
            specDraftCacheDtype:
              pendingLoadConfig?.specDraftCacheDtype ??
              stateBeforeUnload.specDraftCacheDtype,
            ctxCheckpoints:
              pendingLoadConfig?.ctxCheckpoints ??
              stateBeforeUnload.ctxCheckpoints,
            cacheRam: pendingLoadConfig?.cacheRam ?? stateBeforeUnload.cacheRam,
          };
          try {
            // Pre-flight validation avoids unloading a working model for a clearly invalid id.
            const validateNativePathLease = nativePathToken
              ? (await consumeNativePathToken(nativePathToken, "validate-model")).nativePathLease
              : undefined;
            // Validate with the same effective context /load uses.
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
              ? stagedGpuIds
              : loadSelectedGpuIds;
            // The reset below re-baselines gpuLayers to Auto; mirror it here.
            const validateGpuLayers = resetsPerModelSettings
              ? GPU_LAYERS_AUTO
              : loadGpuLayers;
            const validateNParallel = resetsPerModelSettings
              ? (pendingLoadConfig?.nParallel ?? null)
              : loadNParallel;
            const validateReasoningBudget = resetsPerModelSettings
              ? (pendingLoadConfig?.reasoningBudget ?? -1)
              : loadReasoningBudget;
            const validateReasoningBudgetMessage = resetsPerModelSettings
              ? (pendingLoadConfig?.reasoningBudgetMessage ?? "")
              : loadReasoningBudgetMessage;
            const validateNBatch = resetsPerModelSettings
              ? (pendingLoadConfig?.nBatch ?? null)
              : loadNBatch;
            const validateNUbatch = resetsPerModelSettings
              ? (pendingLoadConfig?.nUbatch ?? null)
              : loadNUbatch;
            const validateServerTuning: ServerTuningValues =
              resetsPerModelSettings
                ? {
                    loadMode: pendingLoadConfig?.loadMode ?? null,
                    specDraftCacheDtype:
                      pendingLoadConfig?.specDraftCacheDtype ?? null,
                    ctxCheckpoints: pendingLoadConfig?.ctxCheckpoints ?? null,
                    cacheRam: pendingLoadConfig?.cacheRam ?? null,
                  }
                : loadServerTuning;
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
                loadedContextLength: loadContextLength,
                currentCheckpoint,
                activeGgufVariant: loadActiveGgufVariant,
                isMlx: isServedByMlx(isGguf, platform.deviceType, platform.chatOnlyReason),
                pinnedMaxSeqLength,
                defaultMaxSeqLength: unpinnedDefaultRequest(
                  previousIsMlx,
                  stateBeforeUnload.params.maxSeqLength,
                  DEFAULT_MAX_SEQ_LENGTH,
                ),
                presetSource: loadActivePresetSource,
              }),
            );
            const validation = await validateModel({
              model_path: loadPath,
              engine_precision: stateBeforeUnload.params.enginePrecision ?? "auto",
              engine_parallelism: stateBeforeUnload.params.engineParallelism ?? "tensor",
              engine: isGguf ? "auto" : (stateBeforeUnload.params.engine ?? "auto"),
              nativePathLease: validateNativePathLease,
              hf_token: hfToken,
              max_seq_length: validateMaxSeqLength,
              load_in_4bit: (stateBeforeUnload.params.engine ?? "auto") === "auto",
              is_lora: isLora,
              gguf_variant: ggufVariant ?? null,
              cache_type_kv: loadKvCacheDtype,
              tensor_parallel: loadTensorParallel,
              disable_vision: loadDisableVision,
              gpu_ids: validateGpuIds ?? undefined,
              ...(isGguf
                ? {
                    gpu_memory_mode: loadGpuMemoryMode,
                    // Sized like the follow-up /load, else a manual DiffusionGemma split 409s when it fits.
                    gpu_layers: validateGpuLayers,
                    n_parallel: validateNParallel,
                    reasoning_budget: targetIsDiffusion
                      ? -1
                      : validateReasoningBudget,
                    reasoning_budget_message: targetIsDiffusion
                      ? ""
                      : validateReasoningBudgetMessage,
                    ...(validateNBatch != null
                      ? { n_batch: validateNBatch }
                      : {}),
                    ...(validateNUbatch != null
                      ? { n_ubatch: validateNUbatch }
                      : {}),
                    // Same values as /load: the preflight must approve the command /load sends.
                    ...serverTuningLoadPayload(validateServerTuning),
                    // Same extra args as /load: --ctx-size or cache overrides change the memory estimate.
                    ...(!targetIsDiffusion && loadLlamaExtraArgs !== undefined
                      ? { llama_extra_args: loadLlamaExtraArgs ?? [] }
                      : {}),
                  }
                : {}),
            });
            isLora = validation.is_lora ?? isLora;
            // validateModel cannot be aborted, so stop here if a replacement superseded this load.
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");
            if (validation.mlx_loads_base_model) {
              mlxLoadProgress = true;
              const mlxBaseDescription = isLora
                ? `Loading the adapter with ${validation.mlx_loads_base_model} in place of its bitsandbytes base, downloading it first if needed.`
                : `Loading ${validation.mlx_loads_base_model} instead, downloading it first if needed.`;
              progressModelIds = isLora && !isLocal
                ? [modelId, validation.mlx_loads_base_model]
                : [validation.mlx_loads_base_model];
              downloadComplete = false;
              loadingDescription = [
                currentCheckpoint ? switchingNote : null,
                extraLoadingDescription ?? null,
                mlxBaseDescription,
              ]
                .filter(Boolean)
                .join(" ");
              setLoadProgress({
                percent: 0,
                label: "Preparing download",
                phase: "downloading",
              });
              toast.info("MLX cannot use 4-bit bitsandbytes weights", {
                description: mlxBaseDescription,
              });
            }
            // Upgrade consent runs before the security dialogs; Accept installs and the load continues.
            if (validation.requires_transformers_upgrade) {
              const upgraded = await confirmTransformersUpgradeIfNeeded({
                modelName: modelId,
                upgrade: validation.transformers_upgrade,
                // No installable release: custom-code models may fall back to the trust_remote_code gate.
                trustRemoteCodeFallback: validation.requires_trust_remote_code,
                // The install refuses while chats generate and has no force flag; pass the confirmed cancel.
                forceCancelActive,
              });
              // The install unloads the previous model even on failure, so any later exit must roll back.
              if (
                useTransformersUpgradeDialogStore
                  .getState()
                  .consumeServerUnloadedChat()
                && currentCheckpoint
              ) {
                // The installer may unload the resident; cancellation reads this run-level marker.
                loadRun.residentModelUnloaded = true;
                previousWasUnloaded = true;
              }
              if (!upgraded) {
                throw new Error(getTransformersUpgradeRequiredMessage(displayName));
              }
            }
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");
            // Opens even with trustRemoteCode on: the worker needs a fingerprint only the dialog makes.
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
              });
              if (!approved) {
                throw new Error(getTrustRemoteCodeRequiredMessage(displayName));
              }
            }
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");
            const loadNativePathLease = nativePathToken
              ? (await consumeNativePathToken(nativePathToken, "load-model")).nativePathLease
              : undefined;

            stopQueuedRuns(stopDecision, keepsOthers || touchesOnlySelected);
            if (currentCheckpoint && !keepsOthers) {
              // With chats generating, skip the preliminary unload so a rejected target truncates nothing.
              if (!forceCancelActive && !touchesOnlySelected) {
                await unloadModel({ model_path: currentCheckpoint });
                // Only a real /unload removes the resident; the forced path leaves it to /load.
                loadRun.residentModelUnloaded = true;
              }
              // Set either way: /load can still leave no model resident, and an unneeded rollback hits
              // already_loaded before the gate.
              previousWasUnloaded = true;
            }
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");

            // On a switch, fall back to the standing spec preference so forced MTP does not follow.
            if (resetsPerModelSettings) {
              const persistedSpeculativeType = readPersistedSpeculativeType();
              useChatRuntimeStore.setState({
                speculativeType: persistedSpeculativeType,
                loadedSpeculativeType: persistedSpeculativeType,
                specDraftNMax: null,
                loadedSpecDraftNMax: null,
                nParallel: null,
                loadedNParallel: null,
                reasoningBudget: -1,
                loadedReasoningBudget: null,
                loadedReasoningBudgetRequested: null,
                reasoningBudgetMessage: "",
                loadedReasoningBudgetMessage: null,
                loadedReasoningBudgetMessageRequested: null,
                nBatch: null,
                loadedNBatch: null,
                nUbatch: null,
                loadedNUbatch: null,
                // Cleared in both halves, or a later rollback re-sends the departed model's baseline.
                ...clearedServerTuningState(),
                selectedGpuIds: null,
                selectedGpuIndexKind: null,
                gpuLayers: GPU_LAYERS_AUTO,
                nCpuMoe: 0,
                splitRatio: null,
                customContextLength: null,
              });
              loadSpeculativeType =
                pendingLoadConfig?.speculativeType != null
                  ? normalizeSpeculativeType(pendingLoadConfig.speculativeType)
                  : persistedSpeculativeType;
              loadSpecDraftNMax = pendingLoadConfig?.specDraftNMax ?? null;
              loadNParallel = pendingLoadConfig?.nParallel ?? null;
              loadReasoningBudget = pendingLoadConfig?.reasoningBudget ?? -1;
              loadReasoningBudgetMessage =
                pendingLoadConfig?.reasoningBudgetMessage ?? "";
              loadNBatch = pendingLoadConfig?.nBatch ?? null;
              loadNUbatch = pendingLoadConfig?.nUbatch ?? null;
              loadServerTuning = {
                loadMode: pendingLoadConfig?.loadMode ?? null,
                specDraftCacheDtype:
                  pendingLoadConfig?.specDraftCacheDtype ?? null,
                ctxCheckpoints: pendingLoadConfig?.ctxCheckpoints ?? null,
                cacheRam: pendingLoadConfig?.cacheRam ?? null,
              };
              loadMlxKvQuant = pendingLoadConfig?.mlxKvQuant ?? null;
              loadMlxInt8Prefill = pendingLoadConfig?.mlxInt8Prefill ?? false;
              loadChatTemplateOverride =
                pendingLoadConfig?.chatTemplateOverride?.trim()
                  ? pendingLoadConfig.chatTemplateOverride
                  : null;
              // Keep the click-time snapshot in lock-step with the store reset.
              loadCustomContextLength =
                pendingLoadConfig?.customContextLength ?? null;
              loadSelectedGpuIds = stagedGpuIds;
              loadGpuLayers = pendingLoadConfig?.gpuLayers ?? GPU_LAYERS_AUTO;
              loadNCpuMoe = pendingLoadConfig?.nCpuMoe ?? 0;
              loadSplitRatio = pendingLoadConfig?.tensorSplit ?? null;
            }

            // The user's Context Length, captured before the clamp: this is what the load pins.
            const targetIsMlx = isServedByMlx(
              isGguf,
              platform.deviceType,
              platform.chatOnlyReason,
            );
            const explicitCtxPin = loadRequestContextPin(
              loadCustomContextLength,
              targetIsMlx,
              pinnedMaxSeqLength,
            );
            // Same-model layer pin keeps the resolved context: sending 0 means native under --fit off (OOM).
            if (
              isGguf &&
              !switchingModelOrVariant &&
              loadGpuMemoryMode === "manual" &&
              loadGpuLayers >= 0 &&
              loadCustomContextLength == null &&
              (loadContextLength ?? 0) > 0
            ) {
              loadCustomContextLength = loadContextLength;
            }
            const effectiveMaxSeqLength = resolveLoadMaxSeqLength({
              modelId,
              ggufVariant,
              isGguf,
              customContextLength: loadCustomContextLength,
              loadedContextLength: loadContextLength,
              currentCheckpoint,
              activeGgufVariant: loadActiveGgufVariant,
              isMlx: targetIsMlx,
              pinnedMaxSeqLength,
              defaultMaxSeqLength: unpinnedDefaultRequest(
                  previousIsMlx,
                  stateBeforeUnload.params.maxSeqLength,
                  DEFAULT_MAX_SEQ_LENGTH,
                ),
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
            if (!keepsOthers && !touchesOnlySelected) {
              requestLocalPromptQueueStop();
            }
            if (lifecycleLease !== null) {
              chatModelLifecycleGate.markLoading(lifecycleLease);
            }
            // Bind the request ID before the POST so an early cancel lands as a backend tombstone.
            loadRun.loadAttemptPath = loadPath;

            const loadResponse = await loadModel({
              model_path: loadPath,
              engine_precision: stateBeforeUnload.params.enginePrecision ?? "auto",
              engine_parallelism: stateBeforeUnload.params.engineParallelism ?? "tensor",
              engine: isGguf ? "auto" : (stateBeforeUnload.params.engine ?? "auto"),
              load_request_id: loadRun.requestId,
              nativePathLease: loadNativePathLease,
              hf_token: hfToken,
              max_seq_length: loadMaxSeqLength,
              max_seq_length_auto_derived: isReplayedLoadContext(
                isGguf,
                loadCustomContextLength,
                loadMaxSeqLength,
              ),
              load_in_4bit: (stateBeforeUnload.params.engine ?? "auto") === "auto",
              is_lora: isLora,
              gguf_variant: ggufVariant ?? null,
              trust_remote_code: trustRemoteCode,
              approved_remote_code_fingerprint: approvedRemoteCodeFingerprint,
              chat_template_override: effectiveChatTemplateOverride,
              cache_type_kv: loadKvCacheDtype,
              mlx_kv_quant: loadMlxKvQuant ?? null,
              mlx_int8_prefill: loadMlxInt8Prefill,
              speculative_type: loadSpeculativeType,
              spec_draft_n_max: loadSpecDraftNMax,
              n_parallel: loadNParallel,
              reasoning_budget:
                isGguf && !targetIsDiffusion ? loadReasoningBudget : -1,
              reasoning_budget_message:
                isGguf && !targetIsDiffusion ? loadReasoningBudgetMessage : "",
              // [] means launch with none; llama-server flags only, never transformers or diffusion.
              ...(isGguf && !targetIsDiffusion && loadLlamaExtraArgs !== undefined
                ? { llama_extra_args: loadLlamaExtraArgs ?? [] }
                : {}),
              // omitted when blank: a null counts as set and strips inherited -b / -ub
              ...(isGguf && loadNBatch != null ? { n_batch: loadNBatch } : {}),
              ...(isGguf && loadNUbatch != null
                ? { n_ubatch: loadNUbatch }
                : {}),
              ...(isGguf && !targetIsDiffusion
                ? serverTuningLoadPayload(loadServerTuning)
                : {}),
              tensor_parallel: loadTensorParallel,
              disable_vision: loadDisableVision,
              gpu_memory_mode: loadGpuMemoryMode,
              gpu_layers: loadGpuLayers,
              n_cpu_moe: loadNCpuMoe,
              tensor_split: loadSplitRatio ?? undefined,
              gpu_ids: loadSelectedGpuIds ?? undefined,
              force_cancel_active: forceCancelActive,

              force_reload: forceReload,
              alongside: keepModelsLoaded || touchesOnlySelected,
            });
            cpuFallbackReason = loadResponse.cpu_fallback_reason ?? null;
            mmprojFallbackReason = loadResponse.mmproj_fallback_reason ?? null;
            offloadCounts = offloadCountsFrom(loadResponse);
            if (loadResponse.evicted?.length) {
              toast.info(
                `Unloaded ${loadResponse.evicted.join(", ")} to make room`,
                {
                  description:
                    "Select it again to load it back.",
                },
              );
            }

            if (abortCtrl.signal.aborted) throw new Error("Cancelled");

            // Persist the requested spec intent, not the echo; skipped for per-model configs.
            if (!keepSpeculative) {
              saveSpeculativeType(loadSpeculativeType);
            }
            // Persist only on success so an abandoned selection does not stick.
            persistGpuMemoryModeOnLoad(loadResponse, loadGpuMemoryMode);

            const currentParams = useChatRuntimeStore.getState().params;
            const loadedFields = loadedContextFields(loadResponse);
            const loadedContextCap = replayMaxTokensCap(
              loadedFields.loadedContextLength ??
                (!loadResponse.is_gguf && effectiveMaxSeqLength > 0
                  ? effectiveMaxSeqLength
                  : null),
            );
            setParams(
              {
                ...mergeBackendRecommendedInference({
                  current: currentParams,
                  response: loadResponse,
                  modelId,
                  presetSource: useChatRuntimeStore.getState().activePresetSource,
                  loadedContextLength: loadedFields.loadedContextLength,
                }),
                ...(isGguf
                  ? {}
                  : {
                      maxSeqLength: loadedContextForParams(
                        loadedFields.loadedContextLength,
                        loadMaxSeqLength,
                        currentParams.maxSeqLength,
                      ),
                    }),
              },
              // Remembered settings over defaults, but no budget larger than the loaded context.
              {
                fromModelDefaults: true,
                maxTokensCap: loadedContextCap,
              },
            );
            // Qwen3.5/3.6 small models default thinking off; anchored so "qwen3.5" itself does not match.
            let reasoningDefault = loadResponse.supports_reasoning ?? false;
            if (reasoningDefault) {
              const mid = modelId.toLowerCase();
              if (mid.includes("qwen3.5") || mid.includes("qwen3.6")) {
                // Scan segments right to left; the trailing boundary stops "8bit".
                const sizeRe = /(?:^|[-_.])(\d+\.?\d*)\s*([bm])(?:$|[-_.])/;
                const sizeMatch = mid
                  .replace(/\\/g, "/")
                  .split("/")
                  .reduceRight<RegExpMatchArray | null>(
                    (found, seg) => found ?? seg.match(sizeRe),
                    null,
                  );
                if (sizeMatch) {
                  const size = parseFloat(sizeMatch[1]);
                  const sizeB = sizeMatch[2] === "m" ? size / 1000 : size;
                  if (sizeB <= 9) reasoningDefault = false;
                }
              }
            }
            const loadedKv = loadResponse.cache_type_kv ?? null;
            const loadedTp = loadResponse.tensor_parallel ?? false;
            const loadedSpec = normalizeSpeculativeType(
              loadResponse.speculative_type,
            );
            const committedSlots =
              ((loadResponse.is_gguf ?? false) && !(loadResponse.is_diffusion ?? false)) ||
              (loadResponse.is_mlx ?? false)
                ? (loadNParallel ?? null)
                : null;
            const committedNBatch =
              (loadResponse.is_gguf ?? false) &&
              !(loadResponse.is_diffusion ?? false)
                ? (loadNBatch ?? null)
                : null;
            const committedNUbatch =
              (loadResponse.is_gguf ?? false) &&
              !(loadResponse.is_diffusion ?? false)
                ? (loadNUbatch ?? null)
                : null;
            const committedServerTuning =
              (loadResponse.is_gguf ?? false) &&
              !(loadResponse.is_diffusion ?? false)
                ? committedServerTuningState(loadServerTuning)
                : clearedServerTuningState();
            // Keep the user's pin so an Auto load stays Auto; MLX pins the same way.
            const keepCustomCtx = resolveExplicitCtxPin(
              loadResponse.is_gguf || targetIsMlx ? explicitCtxPin : null,
            );
            const reasoningAlwaysOn = loadResponse.reasoning_always_on ?? false;
            const reasoningStyle = loadResponse.reasoning_style ?? "enable_thinking";
            const supportsReasoning = loadResponse.supports_reasoning ?? false;
            const supportsPreserveThinking =
              loadResponse.supports_preserve_thinking ?? false;
            const supportsTools = loadResponse.supports_tools ?? false;
            const reasoningEffortLevels =
              loadResponse.reasoning_effort_levels &&
              loadResponse.reasoning_effort_levels.length > 0
                ? (loadResponse.reasoning_effort_levels as ReasoningEffort[])
                : (["low", "medium", "high"] as const);
            // A pin's effort level is one model's; use the chat's own level after a switch.
            const existingReasoningEffort =
              (pinHoldsLiveEffort() ? takeEffortDisplacedByPin() : null) ??
              useChatRuntimeStore.getState().reasoningEffort;
            const clampedReasoningEffort =
              reasoningStyle === "enable_thinking_effort" ||
              reasoningStyle === "reasoning_effort"
                ? clampReasoningEffortToLevels(
                    existingReasoningEffort,
                    reasoningEffortLevels,
                  )
                : clampLocalReasoningEffort(existingReasoningEffort);
            const nextReasoningEnabled = reasoningAlwaysOn
              ? true
              : reloadingSameModel && supportsReasoning
                ? stateBeforeUnload.reasoningEnabled
                : reasoningDefault;
            rememberApprovedRemoteCode(modelId, approvedRemoteCodeFingerprint);
            // A later rollback reads the snapshot path, not the id this was stored under.
            rememberApprovedRemoteCode(loadPath, approvedRemoteCodeFingerprint);
            useChatRuntimeStore.setState({
              ...loadedContextFields(loadResponse),
              modelRequiresTrustRemoteCode:
                loadResponse.requires_trust_remote_code ?? false,
              supportsReasoning,
              reasoningAlwaysOn,
              reasoningEnabled: nextReasoningEnabled,
              reasoningStyle,
              supportsReasoningOff: reasoningStyle !== "reasoning_effort",
              reasoningEffortLevels,
              reasoningEffort: clampedReasoningEffort,
              supportsPreserveThinking,
              preserveThinking:
                reloadingSameModel && supportsPreserveThinking
                  ? stateBeforeUnload.preserveThinking
                  : resolvePreserveThinkingOnLoad(loadResponse),
              supportsTools,
              ...(reloadingSameModel && supportsTools
                ? {
                    toolsEnabled: stateBeforeUnload.toolsEnabled,
                    codeToolsEnabled: stateBeforeUnload.codeToolsEnabled,
                  }
                : resolveToolsEnabledOnLoad(supportsTools)),
              kvCacheDtype: loadedKv,
              loadedKvCacheDtype: loadedKv,
              ...mlxRuntimeStateFrom(loadResponse),
              tensorParallel: loadedTp,
              loadedTensorParallel: loadedTp,
              loadedDisableVision: loadResponse.disable_vision ?? false,
              // Take the echo: loadDisableVision forces off for diffusion without writing the store.
              disableVision: loadResponse.disable_vision ?? false,
              loadedVisionDisabledByUser:
                loadResponse.vision_disabled_by_user ?? false,
              ...loadedGpuMemoryFields(loadResponse),
              speculativeType: loadedSpec,
              loadedSpeculativeType: loadedSpec,
              specDraftNMax: loadResponse.spec_draft_n_max ?? null,
              loadedSpecDraftNMax: loadResponse.spec_draft_n_max ?? null,
              // Keep the click-time value: adopting the resolved echo would pin a blank control.
              nParallel: committedSlots,
              loadedNParallel: committedSlots,
              reasoningBudget:
                (loadResponse.is_gguf ?? false) &&
                !(loadResponse.is_diffusion ?? false)
                  ? (loadResponse.reasoning_budget ?? loadReasoningBudget)
                  : -1,
              loadedReasoningBudget:
                (loadResponse.is_gguf ?? false) &&
                !(loadResponse.is_diffusion ?? false)
                  ? (loadResponse.reasoning_budget ?? loadReasoningBudget)
                  : -1,
              reasoningBudgetMessage:
                (loadResponse.is_gguf ?? false) &&
                !(loadResponse.is_diffusion ?? false)
                  ? (loadResponse.reasoning_budget_message ??
                    loadReasoningBudgetMessage)
                  : "",
              loadedReasoningBudgetMessage:
                (loadResponse.is_gguf ?? false) &&
                !(loadResponse.is_diffusion ?? false)
                  ? (loadResponse.reasoning_budget_message ??
                    loadReasoningBudgetMessage)
                  : "",
              loadedReasoningBudgetRequested:
                loadResponse.is_gguf && !loadResponse.is_diffusion
                  ? (loadResponse.requested_reasoning_budget ?? loadReasoningBudget)
                  : -1,
              loadedReasoningBudgetMessageRequested:
                loadResponse.is_gguf && !loadResponse.is_diffusion
                  ? (loadResponse.requested_reasoning_budget_message ?? loadReasoningBudgetMessage)
                  : "",
              nBatch: committedNBatch,
              loadedNBatch: committedNBatch,
              ...committedServerTuning,
              // Omitted fields inherit the resident's list; an explicit [] stays empty, not null.
              loadedLlamaExtraArgs:
                loadResponse.requested_llama_extra_args !== undefined
                  ? (loadResponse.requested_llama_extra_args ?? [])
                  : loadLlamaExtraArgs !== undefined
                    ? (loadLlamaExtraArgs ?? [])
                    : resetsPerModelSettings
                      ? null
                      : (stateBeforeUnload.loadedLlamaExtraArgs ?? null),
              nUbatch: committedNUbatch,
              loadedNUbatch: committedNUbatch,
              customContextLength: keepCustomCtx,
              loadedCustomContextLength: keepCustomCtx,
              defaultChatTemplate: loadResponse.chat_template ?? null,
              chatTemplateOverride: effectiveChatTemplateOverride,
              loadedChatTemplateOverride: effectiveChatTemplateOverride,
              loadedIsMultimodal: isMultimodalResponse(loadResponse),
              mmprojFallbackReason: loadResponse.mmproj_fallback_reason ?? null,
              loadedIsDiffusion: loadResponse.is_diffusion ?? false,
              activeModelIsLocal: loadResponse.is_local_model ?? false,
              activeLoadId: loadPath === modelId ? null : loadPath,
              activeNativePathToken: nativePathToken ?? null,
              activeNativePathExpiresAtMs: nativePathToken
                ? nativePathExpiresAtMs
                : null,
            });
            noteLoadedModelReasoningMode(modelId, nextReasoningEnabled, true);
            syncModelCapabilities(modelId, loadResponse);
            const p = resolveQwenThinkingParams(
              modelId,
              nextReasoningEnabled,
            );
            if (
              p !== null &&
              (loadResponse.supports_reasoning ?? false)
            ) {
              const store = useChatRuntimeStore.getState();
              if (store.activePresetSource === "builtin-default") {
                store.setParams({ ...store.params, ...p }, {
                  fromModelDefaults: true,
                  maxTokensCap: loadedContextCap,
                });
              }
            }
            await refresh({ signal: abortCtrl.signal });
            postLoadRefresh.needed = Boolean(
              (loadResponse.is_gguf || isGguf || ggufVariant) &&
                !isExternalModelId(modelId),
            );
            useChatRuntimeStore.setState({
              loadedEngine: loadResponse.engine ?? "auto",
              loadedEnginePrecision: loadResponse.engine_precision ?? "auto",
              loadedEngineParallelism: loadResponse.engine_parallelism ?? "tensor",
            });
            // Native file-picker paths need an expiring lease, so they are not remembered.
            const indexedLocalPick =
              typeof selection !== "string" && selection.source === "local";
            if (
              !isLora &&
              !(loadResponse.is_lora ?? false) &&
              !nativePathToken &&
              !isExternalModelId(modelId) &&
              (indexedLocalPick || !isLocalModelPath(modelId))
            ) {
              recordLastLocalModelLoad({
                id: modelId,
                kind:
                  loadResponse.is_gguf || isGguf || ggufVariant
                    ? "gguf"
                    : "model",
                ggufVariant: ggufVariant ?? null,
              });
            }
          } catch (error) {
            notifyLocalPromptQueueLoadFailed(lifecycleLease);
            if (abortCtrl.signal.aborted) throw error;
            // An unanswered load may still be running; only roll back an answered failure.
            if (
              previousWasUnloaded &&
              previousCheckpoint &&
              shouldRestorePreviousModel(error)
            ) {
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
                const rollbackResponse = await loadModel({
                  model_path: previousActiveLoadId || previousCheckpoint,
                  engine: stateBeforeUnload.loadedEngine,
                  engine_precision: stateBeforeUnload.loadedEnginePrecision,
                  engine_parallelism: stateBeforeUnload.loadedEngineParallelism,
                  nativePathLease: rollbackNativePathLease,
                  hf_token: hfToken,
                  max_seq_length: rollbackMaxSeqLength,
                  load_in_4bit: stateBeforeUnload.loadedEngine === "auto",
                  is_lora: previousIsLora,
                  gguf_variant: previousVariant,
                  trust_remote_code:
                    previousModelRequiresTrustRemoteCode || trustRemoteCode,
                  // Resend the previous model's pinned approval so restoring it is not re-blocked.
                  approved_remote_code_fingerprint:
                    approvedRemoteCodeFingerprints.get(previousCheckpoint) ?? null,
                  chat_template_override:
                    rollbackState.loadedChatTemplateOverride,
                  cache_type_kv: rollbackState.loadedKvCacheDtype,
                  mlx_kv_quant: rollbackState.loadedMlxKvQuantRequested,
                  mlx_int8_prefill: rollbackState.loadedMlxInt8PrefillRequested,
                  speculative_type:
                    rollbackState.loadedSpeculativeType,
                  spec_draft_n_max:
                    rollbackState.loadedSpecDraftNMax,
                  n_parallel: rollbackState.loadedNParallel,
                  reasoning_budget:
                    rollbackState.loadedReasoningBudgetRequested ?? -1,
                  reasoning_budget_message:
                    rollbackState.loadedReasoningBudgetMessageRequested ?? "",
                  // omit unset fields: a null counts as set and would strip the previous server's extras
                  ...(rollbackState.loadedNBatch != null
                    ? { n_batch: rollbackState.loadedNBatch }
                    : {}),
                  ...(rollbackState.loadedNUbatch != null
                    ? { n_ubatch: rollbackState.loadedNUbatch }
                    : {}),
                  ...serverTuningLoadPayload({
                    loadMode: rollbackState.loadedLoadMode,
                    specDraftCacheDtype:
                      rollbackState.loadedSpecDraftCacheDtype,
                    ctxCheckpoints: rollbackState.loadedCtxCheckpoints,
                    cacheRam: rollbackState.loadedCacheRam,
                  }),
                  // Explicit: the target is resident now, so an omitted field would inherit across models.
                  ...(rollbackState.loadedLlamaExtraArgs != null
                    ? { llama_extra_args: rollbackState.loadedLlamaExtraArgs }
                    : {}),
                  tensor_parallel: rollbackState.loadedTensorParallel ?? false,
                  // The previous server's value; the control already holds the target's setting.
                  disable_vision: rollbackState.loadedDisableVision ?? false,
                  gpu_memory_mode: rollbackState.loadedGpuMemoryMode ?? "auto",
                  gpu_layers: rollbackState.loadedGpuLayers ?? GPU_LAYERS_AUTO,
                  cpu_fallback: rollbackState.loadedCpuFallback,
                  n_cpu_moe: rollbackState.loadedNCpuMoe ?? 0,
                  alongside: keepModelsLoaded || touchesOnlySelected,
                  tensor_split: rollbackState.loadedSplitRatio ?? undefined,
                  gpu_ids: rollbackState.loadedGpuIds ?? undefined,
                  // The failed swap already unloaded the server those runs used.
                  force_cancel_active: true,
                });
                const rollbackSpeculativeType = normalizeSpeculativeType(
                  rollbackResponse.speculative_type,
                );
                useChatRuntimeStore.setState({
                  activeModelIsLocal: rollbackResponse.is_local_model ?? false,
                  activeLoadId: previousActiveLoadId ?? null,
                  activeNativePathToken: previousActiveNativePathToken ?? null,
                  // Restore the lease with the token so token A never pairs with load B's expiry.
                  activeNativePathExpiresAtMs: previousActiveNativePathToken
                    ? (previousActiveNativePathExpiresAtMs ?? null)
                    : null,
                  speculativeType: rollbackState.loadedSpeculativeType ?? null,
                  specDraftNMax: rollbackState.loadedSpecDraftNMax ?? null,
                  nParallel: previousNParallel,
                  loadedNParallel: rollbackState.loadedNParallel ?? null,
                  reasoningBudget: previousReasoningBudget,
                  loadedReasoningBudget:
                    rollbackResponse.reasoning_budget ?? -1,
                  reasoningBudgetMessage: previousReasoningBudgetMessage,
                  loadedReasoningBudgetMessage:
                    rollbackResponse.reasoning_budget_message ?? "",
                  loadedReasoningBudgetRequested:
                    rollbackResponse.requested_reasoning_budget ??
                    rollbackState.loadedReasoningBudgetRequested ??
                    -1,
                  loadedReasoningBudgetMessageRequested:
                    rollbackResponse.requested_reasoning_budget_message ??
                    rollbackState.loadedReasoningBudgetMessageRequested ??
                    "",
                  nBatch: previousNBatch,
                  loadedNBatch: rollbackState.loadedNBatch ?? null,
                  nUbatch: previousNUbatch,
                  loadedNUbatch: rollbackState.loadedNUbatch ?? null,
                  loadMode: previousServerTuning.loadMode ?? null,
                  loadedLoadMode: rollbackState.loadedLoadMode ?? null,
                  specDraftCacheDtype:
                    previousServerTuning.specDraftCacheDtype ?? null,
                  loadedSpecDraftCacheDtype:
                    rollbackState.loadedSpecDraftCacheDtype ?? null,
                  ctxCheckpoints: previousServerTuning.ctxCheckpoints ?? null,
                  loadedCtxCheckpoints:
                    rollbackState.loadedCtxCheckpoints ?? null,
                  cacheRam: previousServerTuning.cacheRam ?? null,
                  loadedCacheRam: rollbackState.loadedCacheRam ?? null,
                  loadedSpeculativeType: rollbackSpeculativeType,
                  loadedSpecDraftNMax:
                    rollbackResponse.spec_draft_n_max ?? null,
                  loadedKvCacheDtype: rollbackResponse.cache_type_kv ?? null,
                  ...mlxRuntimeStateFrom(rollbackResponse),
                  mlxKvQuant: previousMlxKvQuant,
                  mlxInt8Prefill: previousMlxInt8Prefill,
                  loadedChatTemplateOverride:
                    rollbackState.loadedChatTemplateOverride,
                  ...loadedGpuMemoryFields(rollbackResponse),
                  tensorParallel: rollbackResponse.tensor_parallel ?? false,
                  loadedTensorParallel:
                    rollbackResponse.tensor_parallel ?? false,
                  loadedDisableVision:
                    rollbackResponse.disable_vision ?? false,
                  // Not stateBeforeUnload.disableVision, which holds the target's value by now.
                  disableVision: rollbackState.loadedDisableVision ?? false,
                  loadedVisionDisabledByUser:
                    rollbackResponse.vision_disabled_by_user ?? false,
                  customContextLength:
                    rollbackState.loadedCustomContextLength,
                  loadedCustomContextLength:
                    rollbackState.loadedCustomContextLength,
                });
                await refresh();
              } catch {
                // Rollback also failed; surface the original load error below.
              }
            }
            throw error;
          }
        }

        const isCachedLoad = downloadComplete;
        const toastTitle = isCachedLoad ? "Starting model…" : "Downloading model…";
        const modelLoadToastOptions = (description: ReturnType<typeof renderLoadDescription>) => ({
          description,
          duration: Infinity,
          closeButton: true,
          cancel: {
            label: "Cancel",
            onClick: cancelLoading,
          },
          classNames: MODEL_LOAD_TOAST_CLASSNAMES,
          onDismiss: (dismissedToast: { id: string | number }) => {
            if (loadToastIdRef.current !== dismissedToast.id) {
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

        let progressInterval: ReturnType<typeof setInterval> | null = null;
        const expectedBytes =
          typeof selection !== "string" ? selection.expectedBytes ?? 0 : 0;

        // One buffer per phase so a flip cannot price the new phase against the old clock. Seconds.
        const dlSamples: TransferSample[] = [];
        const mmapSamples: TransferSample[] = [];

        function estimate(
          samples: TransferSample[],
          bytes: number,
          total: number,
        ): { rate: number; eta: number; stable: boolean } {
          if (typeof document !== "undefined" && document.hidden) {
            // Hidden tabs clamp this interval to ~1/min, which the estimator misreads as cadence.
            samples.length = 0;
            return { rate: 0, eta: 0, stable: false };
          }
          appendSample(samples, Date.now() / 1000, bytes);
          const stats = computeTransferStats(samples, total);
          return {
            rate: stats.stable ? stats.rateBytesPerSecond : 0,
            eta: stats.stable ? stats.etaSeconds : 0,
            stable: stats.stable,
          };
        }

        function composeProgressLabel(
          dlGb: number,
          totalGb: number,
          bytes: number,
          total: number,
          samples: TransferSample[],
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

  // A "cached" load can still re-download (#9094); only byte movement counts as proof.
        const watchForCacheMiss =
          !managedLoad && isDownloaded && !isLocal && nativePathToken == null && !isOllamaModelId(modelId);
        const cacheMissDescription = [
          currentCheckpoint ? switchingNote : null,
          extraLoadingDescription ?? null,
          CACHE_MISS_DOWNLOAD_DESCRIPTION,
        ]
          .filter(Boolean)
          .join(" ");
        let activeLoadingDescription = loadingDescription;
        let cacheMissWatch = EMPTY_CACHE_MISS_WATCH;
        let cacheMissDownload = false;

        const pollDownload = async () => {
          if (abortCtrl.signal.aborted || !loadingModelRef.current) {
            if (progressInterval) clearInterval(progressInterval);
            return;
          }
          try {
            const progressModelIdsAtRequest = [...progressModelIds];
            const progressResponses =
              ggufVariant && expectedBytes > 0
                ? [
                    await getGgufDownloadProgress(
                      modelId,
                      ggufVariant,
                      expectedBytes,
                      hfToken,
                    ),
                  ]
                : await Promise.all(
                    progressModelIdsAtRequest.map((progressModelId) =>
                      getDownloadProgress(progressModelId, hfToken, mlxLoadProgress),
                    ),
                  );
            if (!loadingModelRef.current) return;
            if (
              progressModelIdsAtRequest.length !== progressModelIds.length ||
              progressModelIdsAtRequest.some(
                (progressModelId, index) =>
                  progressModelId !== progressModelIds[index],
              )
            ) {
              return;
            }
            const allDownloadsComplete = progressResponses.every(
              ({ progress }) => progress >= 1,
            );
            const firstProgress = progressResponses[0];
            if (!firstProgress) return;
            const prog =
              progressResponses.find(
                ({ progress }) => progress > 0 && progress < 1,
              ) ??
              progressResponses.find(
                ({ downloaded_bytes, expected_bytes, progress }) =>
                  downloaded_bytes > 0 &&
                  expected_bytes === 0 &&
                  progress === 0,
              ) ??
              progressResponses.find(({ progress }) => progress < 1) ??
              firstProgress;

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
              // Write state only when the toast is dismissed; otherwise each poll re-renders the chat page.
              if (loadToastDismissedRef.current) {
                setLoadProgress({
                  percent: pct,
                  label: progressLabel,
                  phase: "downloading",
                });
                return;
              }
              toast(null, {
                id: toastId,
                ...modelLoadToastOptions(
                  renderLoadDescription(
                    "Downloading model…",
                    activeLoadingDescription,
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
              const unknownTotalLabel = `${dlGb.toFixed(1)} GB downloaded${rateSuffix}`;
              if (loadToastDismissedRef.current) {
                setLoadProgress({
                  percent: null,
                  label: unknownTotalLabel,
                  phase: "downloading",
                });
              } else {
                toast(null, {
                  id: toastId,
                  ...modelLoadToastOptions(
                    renderLoadDescription(
                      "Downloading model…",
                      activeLoadingDescription,
                      null,
                      unknownTotalLabel,
                    ),
                  ),
                });
              }
            } else if (
              allDownloadsComplete &&
              (hasShownProgress ||
                progressModelIds.some(
                  (progressModelId) => progressModelId !== modelId,
                ))
            ) {
              downloadComplete = true;
              if (loadToastDismissedRef.current) {
                setLoadProgress({
                  percent: 100,
                  label: "Download complete",
                  phase: "starting",
                });
              } else {
                toast(null, {
                  id: toastId,
                  ...modelLoadToastOptions(
                    renderLoadDescription(
                      "Starting model…",
                      hasShownProgress
                        ? "Download complete. Loading the model into memory."
                        : loadingDescription,
                      hasShownProgress ? 100 : null,
                      hasShownProgress ? "Download complete" : null,
                    ),
                  ),
                });
              }
            }
          } catch {
            // Ignore polling errors; keep polling.
          }
        };

        const pollLoad = async () => {
          if (abortCtrl.signal.aborted || !loadingModelRef.current) {
            if (progressInterval) clearInterval(progressInterval);
            return;
          }
          try {
            const prog = await getLoadProgress();
            if (!loadingModelRef.current) return;
            if (!prog || prog.phase == null) return;
            if (prog.phase === "ready") {
              if (progressInterval) clearInterval(progressInterval);
              return;
            }
            if (managedLoad && prog.bytes_total <= 0) {
              const label = prog.phase === "warming_up"
                ? "Warming up inference kernels. The first load can take several minutes."
                : prog.phase === "loading_weights"
                  ? "Loading model weights into GPU memory."
                  : "Starting the inference engine and preparing model files.";
              if (loadToastDismissedRef.current) {
                setLoadProgress({ percent: 0, label, phase: "starting" });
              } else {
                toast(null, { id: toastId, ...modelLoadToastOptions(renderLoadDescription("Starting model...", label)) });
              }
              return;
            }
            if (prog.bytes_total <= 0) return;
            // Decimal GB to match the file size Hugging Face reports.
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
            if (loadToastDismissedRef.current) {
              setLoadProgress({
                percent: pct,
                label,
                phase: "starting",
              });
              return;
            }
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

        const cacheMissDownloadStarted = async (): Promise<boolean> => {
          try {
            const reading = await getDownloadProgress(modelId, hfToken);
              // Re-read after the await: the load can finish or be cancelled in flight.
            if (abortCtrl.signal.aborted || !loadingModelRef.current) return false;
            const verdict = watchCacheMissDownload(cacheMissWatch, reading);
            cacheMissWatch = verdict.watch;
            if (!verdict.started) return false;
            cacheMissDownload = true;
              // pollDownload's completion branch is gated on this flag.
            hasShownProgress = true;
            downloadComplete = false;
            activeLoadingDescription = cacheMissDescription;
            setLoadProgress({
              percent: verdict.percent,
              label: "Downloading the rest of the model",
              phase: "downloading",
            });
            return true;
          } catch {
            return false;
          }
        };

        const pollProgress = async () => {
          if (!downloadComplete) {
            await pollDownload();
            return;
          }
          if (watchForCacheMiss && !cacheMissDownload && (await cacheMissDownloadStarted())) {
            await pollDownload();
            return;
          }
          await pollLoad();
        };

        let hasShownProgress = false;
        setTimeout(pollProgress, 500);
        progressInterval = setInterval(pollProgress, 2000);

        try {
          await performLoad();
          if (abortCtrl.signal.aborted) return;
          const notice = loadFallbackNotice(
            `${toastDisplayName} loaded`,
            cpuFallbackReason,
            mmprojFallbackReason,
            offloadWarning(offloadCounts),
          );
          const loadedTitle = notice.title;
          const loadedDescription = notice.description;
          const showLoadedToast = notice.degraded ? toast.warning : toast.success;
          if (loadToastDismissedRef.current) {
            showLoadedToast(loadedTitle, {
              description: loadedDescription,
              closeButton: true,
              duration: 8000,
            });
          } else {
            showLoadedToast(loadedTitle, {
              id: toastId,
              description: loadedDescription,
              cancel: undefined,
              closeButton: true,
              duration: 8000,
              onDismiss: undefined,
            });
          }
        } catch (err) {
          if (!abortCtrl.signal.aborted) {
            const message =
              err instanceof Error ? err.message : "Failed to load model";
            const [summary, ...rest] = message.split("\n");
            const detail = rest.join("\n").trim();
            const runnerLogPath = failureLogPath(message);
            const logsAction = runnerLogPath
              ? viewLogsAction(
                  loadFailureLogFamily(isGguf, isDiffusion, runnerLogPath),
                  runnerLogPath,
                )
              : undefined;
            if (loadToastDismissedRef.current) {
              toast.error(summary, {
                description: detail || undefined,
                action: logsAction,
              });
            } else {
              toast.error(summary, {
                id: toastId,
                description: detail || undefined,
                action: logsAction,
                cancel: undefined,
                classNames: undefined,
                closeButton: true,
                duration: 8000,
                onDismiss: undefined,
              });
            }
          }
          throw err;
        } finally {
          if (progressInterval) clearInterval(progressInterval);
          resetLoadingUiForRun(loadRun);
          if (postLoadRefresh.needed && !abortCtrl.signal.aborted) {
            void refreshContextUsage({ afterModelLoad: true });
          }
        }
      } catch (error) {
        // A superseded run must not restore over the replacement, which owns restoration now.
        if (modelSelectionIntentEpoch === loadIntentId) restorePreviousConfig();
        if (abortCtrl.signal.aborted) return;
        resetLoadingUiForRun(loadRun);
        const message =
          error instanceof Error ? error.message : "Failed to load model";
        setModelsError(message);
        setLastModelLoadError(message);
        if (throwOnError) {
          throw error instanceof Error ? error : new Error(message);
        }
      } finally {
        // Last act of this run's coroutine: unblocks a cancellation holding the slot.
        markLoadRunSettled();
      }
    },
    [
      cancelLoading,
      loras,
      models,
      params.checkpoint,
      refresh,
      renderLoadDescription,
      resetLoadingUiForRun,
      setLoadToastDismissedState,
      setModelsError,
      setLastModelLoadError,
      setParams,
    ],
  );

  const loadNpuModel = useCallback(
    async (
      modelPath: string,
      reload?: { forceReload?: boolean; config?: PerModelConfig },
    ) => {
      const store = useChatRuntimeStore.getState();
      if (
        !reload?.forceReload &&
        store.params.checkpoint === modelPath &&
        store.residentCheckpoint === modelPath
      ) {
        return;
      }
      const contextLength = savedContextPin(
        reload?.config ?? resolveInitialConfig(modelPath).config,
      );
      if (loadingModelRef.current ?? store.loadingModelPick) {
        toast.info("Another model is already loading", {
          description: "Wait for it to finish or cancel it first.",
        });
        return;
      }
      const lease = store.beginModelLoading("preparing");
      if (lease === null) {
        toast.info("A model is loading", {
          description: "Wait for it to finish or cancel it first.",
        });
        return;
      }
      loadLifecycleLeaseRef.current = lease;
      const loadIntentId = ++modelSelectionIntentEpoch;
      const displayName = modelDisplayName(
        modelPath.slice(modelPath.indexOf(":") + 1),
      );
      const previous = useChatRuntimeStore.getState();
      const previousCheckpoint = previous.params.checkpoint;
      const previousGgufVariant = previous.activeGgufVariant;
      const abortCtrl = new AbortController();
      const signal = abortCtrl.signal;
      let markLoadRunSettled = () => {};
      const settledPromise = new Promise<void>((resolve) => {
        markLoadRunSettled = resolve;
      });
      const loadRun: ActiveModelLoadRun = {
        attemptId: loadIntentId,
        intentId: loadIntentId,
        abortController: abortCtrl,
        requestId: crypto.randomUUID(),
        loadAttemptPath: null,
        cancelPromise: null,
        rollbackCheckpoint: previousCheckpoint,
        rollbackVariant: previousGgufVariant ?? null,
        rollbackLoadId: previous.activeLoadId ?? null,
        rollbackNativePathToken: previous.activeNativePathToken ?? null,
        rollbackNativePathExpiresAtMs:
          previous.activeNativePathExpiresAtMs ?? null,
        rollbackLoadedState: previous,
        residentModelUnloaded: false,
        forceCancelActive: false,
        settledPromise,
        markSettled: markLoadRunSettled,
      };
      activeLoadRunRef.current = loadRun;
      let toastId: string | number | undefined;
      try {
        const stopDecision = await confirmStopRunningChatsIfNeeded(
          "Loading a different model",
        );
        if (!stopDecision.proceed || signal.aborted) return;
        loadAbortRef.current = abortCtrl;
        const loadInfo = {
          id: modelPath,
          displayName,
          isDownloaded: true,
          isCachedLora: false,
          ggufVariant: null,
          nativePathToken: null,
        };
        setModelsError(null);
        setLastModelLoadError(null);
        setLoadingModel(loadInfo);
        loadingModelRef.current = loadInfo;
        useChatRuntimeStore.getState().setLoadingModelPick(pickOf(loadInfo));
        setLoadProgress({ percent: null, label: null, phase: "starting" });
        toastId = toast.loading(`Loading ${displayName} on the NPU`);
        loadToastIdRef.current = toastId;
        loadRun.forceCancelActive = stopDecision.forceCancelActive;
        loadRun.loadAttemptPath = modelPath;
        cancelPreStreamRunReservations(stopDecision.preStreamRunTokens);
        requestLocalPromptQueueStop(stopDecision.promptQueueThreadIds);
        await loadModel({
          model_path: modelPath,
          hf_token: null,
          max_seq_length: contextLength ?? 0,
          load_in_4bit: false,
          is_lora: false,
          force_reload: reload?.forceReload === true,
          force_cancel_active: stopDecision.forceCancelActive,
          load_request_id: loadRun.requestId,
        });
        if (signal.aborted) return;
        const status = await getInferenceStatus();
        if (signal.aborted) return;
        useChatRuntimeStore.getState().setCheckpoint(modelPath, null);
        applyActiveModelStatusToStore(status, {
          previousCheckpoint,
          previousGgufVariant,
          seedLoadParams: true,
        });
        syncModelCapabilities(modelPath, status);
        void refreshContextUsage({ afterModelLoad: true });
        toast.success(`${displayName} loaded on the NPU`, {
          id: toastId,
          closeButton: true,
          duration: 4000,
        });
      } catch (error) {
        if (signal.aborted) return;
        const message =
          error instanceof Error ? error.message : "Failed to load model";
        setModelsError(message);
        setLastModelLoadError(message);
        toast.error(message, {
          id: toastId,
          closeButton: true,
          duration: 8000,
        });
        await syncInferenceStatusToStore().catch(() => {});
      } finally {
        resetLoadingUiForRun(loadRun);
        markLoadRunSettled();
      }
    },
    [resetLoadingUiForRun, setLastModelLoadError, setModelsError],
  );

  const ejectModel = useCallback(async (
    modelId?: string,
    confirmed?: StopRunningChatsDecision,
  ): Promise<boolean> => {
    if (modelId && modelId !== params.checkpoint) {
      const toastId = toast.loading("Unloading model");
      try {
        if (!(await unloadKeptModel(modelId))) {
          toast.dismiss(toastId);
          return false;
        }
        await refresh();
        toast.success("Model unloaded", { id: toastId, duration: 1200 });
        return true;
      } catch (err) {
        toast.error(
          err instanceof Error ? err.message : "Failed to unload model",
          { id: toastId },
        );
        return false;
      }
    }
    if (!params.checkpoint) {
      return false;
    }
    const bailIfLoading = (): boolean => {
      const runtime = useChatRuntimeStore.getState();
      if (!runtime.modelLoading && !runtime.loadingModelPick) return false;
      if (chatModelLifecycleGate.currentPhase() === "unloading") {
        toast.info("Wait for the model to finish unloading.");
        return true;
      }
      toast.info("A model is loading", {
        description: "Wait for it to finish or cancel it first.",
      });
      return true;
    };
    if (bailIfLoading()) return false;
    setModelsError(null);
    if (isExternalModelId(params.checkpoint)) {
      clearCheckpoint();
      await refresh();
      return true;
    }
    let lifecycleLease: ModelLifecycleLease | null = null;
    try {
      // Hold the lifecycle lease through confirmation and unloading.
      lifecycleLease = useChatRuntimeStore
        .getState()
        .beginModelLoading("unloading");
      if (lifecycleLease === null) {
        return false;
      }
      // Before the running-chats check, which open chat streams can queue (#10339).
      const toastId = toast.loading("Unloading model", {
        description: "Checking for running chats.",
      });
      // Eject stops chats too but leaves no model loaded, so word it as an unload.
      const scope =
        !confirmed && useChatRuntimeStore.getState().loadedModels.length > 1
          ? params.checkpoint
          : undefined;
      const stopDecision =
        confirmed ??
        (await confirmStopRunningChatsIfNeeded(
          "Unloading the model",
          "unload",
          scope,
        ));
      if (!stopDecision.proceed) {
        toast.dismiss(toastId);
        return false;
      }

      async function performUnload(): Promise<void> {
        stopQueuedRuns(stopDecision, Boolean(scope));
        await unloadModel({
          model_path: params.checkpoint,
          force_cancel_active: stopDecision.forceCancelActive,
        });
        if (!scope) {
          requestLocalPromptQueueStop();
        }
        clearCheckpoint();
        await refresh();
      }

      const unloadPromise = performUnload();
      toast.promise(unloadPromise, {
        id: toastId,
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
    } finally {
      if (lifecycleLease !== null) {
        useChatRuntimeStore.getState().endModelLoading(lifecycleLease);
      }
    }
  }, [clearCheckpoint, params.checkpoint, refresh, setModelsError]);

  const ejectAllModels = useCallback(async (): Promise<boolean> => {
    const others = useChatRuntimeStore
      .getState()
      .loadedModels.map((m) => m.checkpoint ?? m.id)
      .filter((id) => id !== params.checkpoint);
    const selectedLocal =
      Boolean(params.checkpoint) && !isExternalModelId(params.checkpoint);
    const decision = await confirmStopRunningChatsIfNeeded(
      "Unloading every model",
      "unload",
    );
    if (!decision.proceed) return false;
    // Before any unload, so a queued send cannot hold a kept model or load one back.
    stopQueuedRuns(decision, false);
    // Others first: the selected model's eject refreshes, which would adopt one still loaded.
    const results = await Promise.allSettled(
      others.map((id) =>
        unloadModel({ model_path: id, force_cancel_active: decision.forceCancelActive }),
      ),
    );
    if (selectedLocal && !(await ejectModel(undefined, decision))) return false;
    await refresh();
    const failed = results.find(
      (result): result is PromiseRejectedResult => result.status === "rejected",
    );
    if (failed) {
      setModelsError(
        failed.reason instanceof Error
          ? failed.reason.message
          : "Failed to unload every model",
      );
      return false;
    }
    return true;
  }, [ejectModel, params.checkpoint, refresh, setModelsError]);

  return {
    refresh,
    selectModel,
    loadNpuModel,
    ejectModel,
    ejectAllModels,
    cancelLoading,
    cancelLoadingForReplacement,
    invalidatePendingModelSelection,
    discardExternalReplacement,
    restoreConfigForExternalReplacement,
    isModelSelectionIntentCurrent,
    loadingModel,
    loadProgress,
    loadToastDismissed,
  };
}
