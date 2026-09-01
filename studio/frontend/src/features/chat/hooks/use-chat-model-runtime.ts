// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { mlxRuntimeStateFrom } from "../lib/mlx-runtime-state";
import {
  type ServerTuningValues,
  clearedServerTuningState,
  committedServerTuningState,
  serverTuningLoadPayload,
} from "../lib/server-tuning-fields";
import { createElement, useCallback, useEffect, useRef, useState } from "react";
import { toast } from "@/lib/toast";
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
import { isSettingsRouteAbsent } from "@/features/settings/api/settings-route-absent";
import { loadModelMemorySettings } from "@/features/settings/api/model-memory";
import { loadVramBudgetSettings } from "@/features/settings/api/vram-budget";
import { loadOpenAIAutoSwitchSettings } from "@/features/settings";
import {
  confirmTransformersUpgradeIfNeeded,
  useTransformersUpgradeDialogStore,
} from "@/features/transformers-upgrade";
import { consumeNativePathToken } from "@/features/native-intents/api";
// eslint-disable-next-line no-restricted-imports -- Avoid the hub barrel's React and download-manager exports.
import { modelDisplayName } from "@/features/hub/lib/model-identity";
// eslint-disable-next-line no-restricted-imports -- Avoid the hub barrel's React and download-manager exports.
import { subscribeResidentStatusRefresh } from "@/features/hub/lib/resident-status-refresh";
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
import { confirmStopRunningChatsIfNeeded } from "../utils/confirm-stop-running-chats";
import { requestLocalPromptQueueStop } from "../utils/prompt-queue-boundary";
import { cancelPreStreamRunReservations } from "../utils/pre-stream-run-reservation";
import type { ModelLifecycleLease } from "../utils/model-lifecycle-gate";
import {
  GPU_LAYERS_AUTO,
  isLocalModelPath,
  loadedGpuMemoryFields,
  persistGpuMemoryModeOnLoad,
  readPersistedGpuMemoryMode,
  readPersistedSpeculativeType,
  reconcilePersistedGpuIds,
  resolvePreserveThinkingOnLoad,
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
  unpinnedDefaultRequest,
  unpinnedLoadContext,
  resolveLoadMaxSeqLength,
  resolveExplicitCtxPin,
  loadRequestContextPin,
  replayMaxTokensCap,
} from "../presets/preset-policy";
import { recordLastLocalModelLoad } from "../utils/last-local-model-load";
import { loadFallbackNotice } from "../utils/mmproj-fallback";
import { resolveQwenThinkingParams } from "../utils/qwen-params";
import { refreshContextUsage } from "../utils/refresh-context-usage";
import { ensureGpuDeviceCache } from "@/hooks/use-gpu-info";
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

export type SelectedModelInput = {
  id: string;
  /** Sent as model_path in place of the id, which stays the identity the UI shows. */
  loadId?: string | null;
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
  /** Staged metadata confirmed the separate DiffusionGemma runner. */
  isDiffusion?: boolean;
  throwOnError?: boolean;
  /** Keep the current speculative-decoding choice across the model switch
   *  instead of resetting it to the standing preference. */
  keepSpeculative?: boolean;
  config?: PerModelConfig;
  previousConfig?: PerModelConfig;
};

// Approved fingerprints by checkpoint, so a rollback after a failed switch can resend
// the pinned approval the worker requires instead of being blocked.
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

/** The two settings reads `serverWideReloadRequired` judges, fetched together. */
async function readServerWideReloadHints(): Promise<boolean> {
  const [modelMemory, vramBudget] = await Promise.all([
    // Forced, like the budget below: a read that started before a save answers about
    // the policy it replaced, and a false from that would suppress the very reload the
    // save was made for.
    readReloadHint(() => loadModelMemorySettings({ force: true })),
    // rethrow, or a failed read is indistinguishable from an absent route: this reader
    // answers null for both, which every other caller is right to treat alike.
    readReloadHint(() =>
      loadVramBudgetSettings({ force: true, rethrow: true }),
    ),
  ]);
  return serverWideReloadRequired({ modelMemory, vramBudget });
}

/** Placement is a set: the backend narrows and reorders it at fit time. */
function sameGpuSelection(
  left: readonly number[] | null | undefined,
  right: readonly number[] | null | undefined,
): boolean {
  const a = [...(left ?? [])].sort((x, y) => x - y);
  const b = [...(right ?? [])].sort((x, y) => x - y);
  return a.length === b.length && a.every((id, index) => id === b[index]);
}

const approvedRemoteCodeFingerprints = new Map<string, string>();
function rememberApprovedRemoteCode(
  checkpoint: string,
  fingerprint: string | null,
): void {
  if (fingerprint) approvedRemoteCodeFingerprints.set(checkpoint, fingerprint);
}

// Class carries the progress-bar spacing; layout is shared CSS now.
const MODEL_LOAD_TOAST_CLASSNAMES = {
  toast: "chat-model-load-toast",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
  cancelButton:
    "!h-auto !rounded-none !border-0 !bg-transparent !px-1 !text-ui-11 !font-normal !text-muted-foreground hover:!bg-transparent hover:!text-destructive focus-visible:!text-destructive",
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
    // A locally scanned model is in no /api/models/list row, so this is the only place
    // its summary learns it is served by MLX, and the seed gate reads that.
    isMlx: Boolean(resp.is_mlx),
    isAudio: Boolean(resp.is_audio),
    audioType: resp.audio_type ?? null,
    hasAudioInput: Boolean(resp.has_audio_input),
    // /api/models/list omits this for the active GGUF row, so without it the
    // video adapter reads false after every load and status hydration.
    hasVideoInput: Boolean(resp.has_video_input),
  };
  const idx = models.findIndex((m) => m.id === modelId);
  if (idx === -1) {
    store.setModels([
      ...models,
      {
        id: modelId,
        // Label like the catalog entry that replaces this on the next /api/models/list;
        // the fallback keeps a cached GGUF's snapshot path out of the bar.
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
    audioType: lora.audio_type ?? null,
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
// Bumped by every call. Two refreshes can be in flight at once -- a media load
// announces itself before its POST and again once the backend has committed the
// eviction -- and they read the status at different moments. Completion order is
// not the order they were issued in, so without this the older one could land
// last and re-pin the model the newer one had just seen released, leaving chat
// claiming a model that 400s on send until the load finally settled.
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

async function syncInferenceStatusToStore(options?: {
  signal?: AbortSignal;
  includeLoras?: boolean;
  preserveIdleUnloaded?: boolean;
}): Promise<void> {
  const signal = options?.signal;
  const includeLoras = options?.includeLoras ?? true;
  // Last issued wins: it read the freshest status, whichever answers first.
  const generation = ++syncGeneration;
  const loraGeneration = includeLoras ? ++loraSyncGeneration : null;
  const superseded = () => generation !== syncGeneration;
  const loraSuperseded = () =>
    loraGeneration !== null && loraGeneration !== loraSyncGeneration;
  const { setModels, setLoras, setCheckpoint, setModelsError } =
    useChatRuntimeStore.getState();
  setModelsError(null);
  try {
    const [listRes, statusRes, , idleUnloadArmed] = await Promise.all([
      listModels(),
      getInferenceStatus(),
      // Settled from this request alone. Read out of the aggregate below, a sibling
      // rejection discarded a good list yet still marked the inventory settled, so a
      // resident LoRA classified as a base model and pinned a new pair generalized.
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

    // Cancellation can land while the requests above are in flight. Bail
    // before writing backend state back -- cancelLoading already cleared it.
    // Same for a refresh that a later one has already superseded: its answer
    // describes a moment that has passed.
    if (signal?.aborted || superseded()) return;

    setModels(listRes.models.map(toChatModelRow));

    const selectedCheckpoint = useChatRuntimeStore.getState().params.checkpoint;
    const isExternalSelectionActive = isExternalModelId(selectedCheckpoint);
    // The local selection is re-derived from this status on every mount, so adopting a
    // TTS model made it the chat model without the user picking it. Read the slot as
    // empty and the eviction branch below clears the stale pick, as it does for an
    // image or video load.
    const chatActiveModel =
      statusRes.active_model && !isSpeechOnlyStatus(statusRes);
    if (chatActiveModel && !isExternalSelectionActive) {
      const checkpointId = resolveInferenceCheckpointId(statusRes);
      if (checkpointId) {
        const previousGgufVariant =
          useChatRuntimeStore.getState().activeGgufVariant;
        // A model loaded outside this tab replaces the resident one, and the pin belonged to the
        // old one: keeping it reloads that one from another's settings.
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
        });
        // setModels(listRes...) above used catalog data, which omits audio
        // capability. Re-apply live status so attach gates survive a refresh.
        syncModelCapabilities(checkpointId, statusRes);

        // Against an already-resident model, history can load before this first status
        // has a checkpoint, so its own recount never runs. The thread guard stops a null
        // thread publishing an empty count.
        const hydrated = useChatRuntimeStore.getState();
        if (
          !selectedCheckpoint &&
          hydrated.contextUsage == null &&
          hydrated.activeThreadId != null &&
          hydrated.loadedContextLength != null &&
          !isExternalModelId(checkpointId)
        ) {
          void refreshContextUsage({ threadId: hydrated.activeThreadId });
        }
      }
    } else if (!chatActiveModel && !isExternalSelectionActive) {
      if (isIdleUnloadedStatus(statusRes, idleUnloadArmed)) return;
      // Loading an image, video or audio model evicts the chat one (the GPU arbiter
      // allows a single owner), and nothing else here would say so: the picker
      // keeps the selection, so the header would go on claiming it is loaded
      // and the next prompt would come back a bare 400.
      const { residentCheckpoint: wasResident, modelLoading } =
        useChatRuntimeStore.getState();
      // specFallbackReason survives here, so clearing activeModelIsLocal
      // alone would flip a local model's warning to "download failed". Every
      // load path and clearCheckpoint set it, so leave it consistent.
      useChatRuntimeStore.setState({
        residentCheckpoint: null,
        modelRequiresTrustRemoteCode: false,
        loadedIsMultimodal: false,
        loadedVisionDisabledByUser: null,
        loadedIsDiffusion: false,
      });
      // A known prior resident or a definitive speech-only owner both prove the
      // persisted Chat pick is stale. A load in flight has no active model yet
      // either, so never clear while one is starting.
      if (
        (wasResident || isSpeechOnlyStatus(statusRes)) &&
        selectedCheckpoint &&
        !modelLoading
      ) {
        // Already the clean id the header shows: resolveInferenceCheckpointId
        // put it there, not a load path.
        if (wasResident) {
          toast.info(`${wasResident} is no longer loaded`, {
            description:
              "The server released it, which loading an image, video or audio model does. Pick it again to keep chatting.",
          });
        }
        // Drop the pick too, which is what a server-side unload already does.
        // Dimming the tick was not enough: the name alone reads as "this is my
        // model" whatever the tick does, and sending to it returns a bare 400.
        useChatRuntimeStore.getState().clearCheckpoint();
      }
    }
  } catch (error) {
    // A superseded refresh reports nothing, or a stale failure would raise a
    // toast about a read whose answer would have been discarded anyway. The LoRA
    // inventory settles from its own request above, never from a sibling's failure.
    if (signal?.aborted || superseded()) return;
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
  // Both llama.cpp update paths land here, and an update replaces the binary whose
  // --help the flag catalogue describes.
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
    (options?: {
      signal?: AbortSignal;
      includeLoras?: boolean;
      preserveIdleUnloaded?: boolean;
    }) =>
      syncInferenceStatusToStore(options),
    [],
  );

  // Nothing here polls /status: refresh runs on mount and when the model lists
  // change, never on a timer. So an image or video load evicting the chat model
  // (the GPU arbiter allows one owner) went unseen, and the header went on
  // showing the evicted model as loaded until something else happened to
  // refresh. Re-read whenever another runtime finishes a load.
  useEffect(
    () =>
      subscribeModelLifecycle(({ runtime }) => {
        // Dictation is a sidecar and evicts nothing; chat's own loads reconcile
        // themselves. A "tts" load does neither: it is the Audio page taking this very
        // slot, so read it back like an image or video load.
        if (runtime === "chat" || runtime === "stt") return;
        // Both edges, not only the settle. The arbiter evicts chat inside the
        // image or video load POST, before the download starts, so waiting for
        // the load to finish left the picker and the header naming a model that
        // had already gone and that 400s on send, for the whole download.
        void refresh({ includeLoras: false });
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

  const cancelLoading = useCallback(() => {
    const model = loadingModelRef.current;
    if (!model) return;
    loadAbortRef.current?.abort();
    loadAbortRef.current = null;
    loadingModelRef.current = null;
    useChatRuntimeStore.getState().clearLoadingModelPick(pickOf(model));
    const tid = loadToastIdRef.current;
    loadToastIdRef.current = null;
    setLoadingModel(null);
    setLoadProgress(null);
    setLoadToastDismissedState(false);
    clearCheckpoint();
    if (tid != null) toast.dismiss(tid);
    const isCachedOrLocal = model.isDownloaded || model.isCachedLora;
    toast.info("Stopped loading model", {
      description: isCachedOrLocal
        ? undefined
        : "The current download may still finish in the background.",
    });
    cancelUnloadPendingRef.current = true;
    void (async () => {
      try {
        // Unforced on purpose: a chat may stream on the PREVIOUS model and must not be killed by
        // cancelling this load. Nothing to report, since the route runs its stop-loading fast
        // path ahead of the active-chat refusal.
        await unloadModel({ model_path: model.id }).catch(() => {});
        // clearCheckpoint above assumed nothing was left loaded, but a forced switch keeps the
        // previous model resident until /load's teardown, and the stop-loading fast path leaves
        // it there. Take the answer from the backend, which reports none once it was evicted.
        await syncInferenceStatusToStore().catch(() => {});
      } finally {
        cancelUnloadPendingRef.current = false;
        if (!loadingModelRef.current) {
          const lease = loadLifecycleLeaseRef.current;
          loadLifecycleLeaseRef.current = null;
          if (lease !== null) {
            useChatRuntimeStore.getState().endModelLoading(lease);
          }
        }
      }
    })();
  }, [clearCheckpoint, setLoadToastDismissedState]);

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
      const restorePreviousConfig = () => {
        if (typeof selection !== "string" && selection.previousConfig) {
          applyPerModelConfigToRuntime(selection.previousConfig, {
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
      if (!forceReload && (!modelId || (params.checkpoint === modelId && (ggufVariant ?? null) === (currentVariant ?? null)))) {
        restorePreviousConfig();
        return;
      }
      // A load is already in flight. If it's this exact pick (id + variant + token),
      // ignore the duplicate click. If it's a DIFFERENT model (including a different
      // GGUF variant of the same repo, which the old id+token guard wrongly treated
      // as a duplicate), don't start a second concurrent load and don't swallow the
      // request: surface it so the user waits or cancels. Centralized here so every
      // entry point is covered, not just the staged Load button.
      const bailIfLoadInFlight = (): boolean => {
        const inFlightLoad =
          loadingModelRef.current ??
          useChatRuntimeStore.getState().loadingModelPick;
        if (!inFlightLoad) return false;
        // The helper form, not an inline apply: it also carries the loaded
        // diffusion flag, which the restored config needs to stay correct.
        restorePreviousConfig();
        const loadingSamePick =
          inFlightLoad.id === modelId &&
          (inFlightLoad.ggufVariant ?? null) === (ggufVariant ?? null) &&
          (inFlightLoad.nativePathToken ?? null) === (nativePathToken ?? null);
        if (loadingSamePick) return true;
        const message =
          "Another model is already loading. Wait for it to finish or cancel it first.";
        setModelsError(message);
        if (throwOnError) throw new Error(message);
        toast.info("Another model is already loading", {
          description: "Wait for it to finish or cancel it first.",
        });
        return true;
      };
      if (bailIfLoadInFlight()) return;

      // Ask the backend, not params.checkpoint: an external pick leaves the local model
      // resident, and a pinned cached row loads under a name its picker row never shows.
      // A staged config always carries forceReload, so Apply still reloads and prompts.
      const selectedCheckpoint =
        useChatRuntimeStore.getState().params.checkpoint;
      const pendingConfig =
        typeof selection !== "string" ? selection.config : undefined;
      // nativePathToken is excluded: a leased file is named by a label two files can share,
      // and only a completed load writes the lease, so adopting would keep a stale token.
      if (!forceReload && !nativePathToken) {
        const residentStatus = await getInferenceStatus().catch(() => null);
        // Warm before reconciling the remembered GPU pick below: load-on-selection can run
        // before any GPU hook mounted, and a cold cache passes the pick through unvalidated
        // while performLoad, which warms it first, would have dropped it.
        if (residentStatus && pendingConfig?.selectedGpuIds !== undefined) {
          await ensureGpuDeviceCache().catch(() => {});
        }
        // What /load would carry for a pick with no saved config, which is not always the
        // live runtime: performLoad treats a different checkpoint or variant as a model
        // switch and clears the per-model fields first (resetsPerModelSettings), so on that
        // door the request is the defaults, and comparing the outgoing model's settings
        // would both prompt for a load that dedupes and adopt one that does not.
        // gpuMemoryMode is standing and kvCacheDtype and tensorParallel are not reset, so
        // those three still come from the store.
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
        // The slot count an unset --parallel resolves to. Session-cached, and 0 is the
        // catalogue's own "unknown", which the comparison reads as a reload.
        const managedFlags = residentStatus
          ? await loadManagedLlamaFlags().catch(() => null)
          : null;
        // Hoisted so it can run twice: everything above was awaited, and in that window
        // another tab can swap the resident model, which this one is never told about
        // (the lifecycle events are dispatched on its own window).
        const adoptable = (status: InferenceStatusResponse) =>
          residentModelMatchesPick(status, {
            id: modelId,
            loadPath,
            ggufVariant,
          }) &&
          // _reuse_loaded_gguf refuses its own already-loaded answer while the audio probe
          // is outstanding, so load_model reaches its fast path and re-probes there.
          // Nothing else does, so skipping /load would leave the model's audio
          // capabilities undetected for as long as the server runs.
          status.audio_probe_pending !== true &&
          // A repairable speculative fallback is the one case where an identical request is
          // not a no-op: _runtime_matches_intent deliberately answers False for it so the
          // next load retries the drafter. Short-circuiting would suppress that and strand
          // the user with speculation off, so let /load through and let the backend decide.
          !residentSpeculativeNeedsRepair(
            status,
            normalizeSpeculativeType(pendingConfig?.speculativeType) ??
              readPersistedSpeculativeType(),
            // The route derives gguf_path from the identifier alone, and the
            // drafter_not_found retry is guarded on its absence.
            (loadPath ?? modelId).toLowerCase().endsWith(".gguf"),
          ) &&
          // The id names the weights, not how the server was invoked. A remembered context
          // length, drafter, placement or extra arg the resident load does not run is a real
          // reload, and skipping it would drop the setting with nothing on screen to say so.
          // Not `pendingConfig`: with no saved record the caller has already run
          // applyModelLoadConfigToRuntime(null), which resets the store to
          // DEFAULT_PER_MODEL_CONFIG, and performLoad reads the store for every field the
          // config does not carry. The live store is therefore what /load would send, on
          // both doors: reset to defaults after a hub or handoff pick, and still the
          // resident values where nothing reset it.
          residentRuntimeMatchesConfig(status, comparedConfig, {
            // What the applier fills an unset field with, so the comparison is against
            // what /load would send rather than against silence.
            speculativeType: readPersistedSpeculativeType(),
            gpuMemoryMode: readPersistedGpuMemoryMode(),
            gpuLayers: GPU_LAYERS_AUTO,
            nCpuMoe: 0,
            // The n_ctx /load would send, from performLoad's own inputs, so an unset
            // length resolves the same way on both sides.
            resolveContextLength: (customContextLength) => {
              const live = useChatRuntimeStore.getState();
              const platform = usePlatformStore.getState();
              // Identity matched above, so the resident model is this pick.
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
            // Never a config field, so the store is the only place it can come from,
            // and the reset clears it before the load reads it.
            splitRatio: resetsPerModelSettings
              ? null
              : useChatRuntimeStore.getState().splitRatio,
            // What /load sends: a pick saved in another index namespace, or naming GPUs
            // that are gone, is reconciled to Automatic before it leaves.
            reconcileGpuIds: (ids, savedIndexKind) =>
              reconcilePersistedGpuIds(
                ids,
                savedIndexKind,
                status.is_diffusion ?? false,
              ),
            normalizeSpeculative: normalizeSpeculativeType,
          });
        if (
          residentStatus &&
          adoptable(residentStatus) &&
          // Both are server-wide, so a save leaves the pick and its config identical while
          // adopt_load_intent_if_matched forces a reload anyway. Without them a policy or
          // budget change made between two picks of one model would never reach the child.
          !(await readServerWideReloadHints())
        ) {
          // Read again, and judge again. A status fetched before those awaits describes the
          // model that was resident then, and adopting it would leave the picker naming
          // this model while prompts went to the one now loaded. A read that fails, or a
          // verdict that no longer holds, falls through to /load, where a real
          // disagreement belongs.
          const confirmedStatus = await getInferenceStatus().catch(() => null);
          if (confirmedStatus && adoptable(confirmedStatus)) {
            // Same window as the confirm below: a rival load may have started during that GET,
            // and it owns the resident model now.
            if (bailIfLoadInFlight()) return;
            // Roll back the config pre-applied for the load that is not happening, before
            // hydrating, so the resident status wins over the staged snapshot. The helper form
            // carries the loaded diffusion flag a resident image model needs.
            restorePreviousConfig();
            // ...except for maxSeqLength, which the rollback has the last word on: it is a
            // client-side generation cap, no status field echoes it, and applyActiveModelStatus
            // below therefore cannot correct it. Leaving it rolled back would hand this model
            // the OUTGOING one's cap, which is visible in truncation but nowhere on screen.
            // An absent cap is not silence either: applyPerModelConfigToRuntime resolves it to
            // defaultInferenceParams.maxSeqLength, so leaving it unset kept the outgoing cap.
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
            // Adopt this pick's own pin, by the rule a completed load writes it: the poll skips
            // its clearing while an external pick is active, so a pin taken for an earlier
            // resident would survive and Apply would reload that old model.
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
            // The pick's own GPU selection, which the hydration above would otherwise
            // widen back. adopt_load_intent_if_matched records the incoming pool when it
            // adopts on the fitted subset (_record_matching_gpu_request), and skipping
            // /load skips that, so the status still names the GPUs the user removed.
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
            // setCheckpoint above blanked the bar, this path returns before the post-load recount,
            // and a mounted thread does not rerun its history loader, so the bar would stay empty.
            void refreshContextUsage({ afterModelLoad: true });
            return;
          }
        }
      }

      // Block queue materialization before taking the cancellation snapshot.
      // A queue that appears while the dialog is open must not be stopped
      // without having been included in the user's confirmation.
      const lifecycleLease = useChatRuntimeStore
        .getState()
        .beginModelLoading();
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

      // Every chat decodes on the llama-server this load replaces, so ask first, then allow the
      // cancel; the 409 gate stays armed for callers that never confirmed.
      let stopDecision: Awaited<
        ReturnType<typeof confirmStopRunningChatsIfNeeded>
      >;
      try {
        stopDecision = await confirmStopRunningChatsIfNeeded(
          forceReload ? "Applying these settings" : "Loading a different model",
        );
      } catch (error) {
        releasePreflightLifecycleLease();
        throw error;
      }
      if (!stopDecision.proceed) {
        releasePreflightLifecycleLease();
        if (typeof selection !== "string" && selection.previousConfig) {
          applyPerModelConfigToRuntime(selection.previousConfig);
        }
        return;
      }
      // Re-check the tracked picker for a load that was already starting when
      // this lifecycle lease was acquired.
      try {
        if (bailIfLoadInFlight()) {
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
      useChatRuntimeStore.getState().setLoadingModelPick(pickOf(loadInfo));
      setLoadProgress(
        isDownloaded || isCachedLora
          ? { percent: null, label: null, phase: "starting" }
          : { percent: 0, label: "Preparing download", phase: "downloading" },
      );
      loadingModelRef.current = loadInfo;
      const abortCtrl = new AbortController();
      loadAbortRef.current = abortCtrl;
      const postLoadRefresh = { needed: false };
      let cpuFallbackReason: CpuFallbackReason | null = null;
      let mmprojFallbackReason: MmprojFallbackReason | null = null;
      try {
        async function performLoad(): Promise<void> {
          if (abortCtrl.signal.aborted) throw new Error("Cancelled");
          let previousWasUnloaded = false;
          const pendingLoadConfig =
            typeof selection !== "string" ? selection.config : undefined;
          // The outgoing model's slot INTENT (blank = follow the server
          // default), which the resolved baseline cannot express. previousConfig
          // is the snapshot the picker took before pre-applying the target's
          // config, so the live control is only the outgoing one without it.
          // Read before the staged-metadata await below so an in-flight change
          // cannot perturb the outgoing snapshot.
          const previousNParallel =
            typeof selection !== "string" && selection.previousConfig
              ? (selection.previousConfig.nParallel ?? null)
              : useChatRuntimeStore.getState().nParallel;
          const previousNBatch =
            typeof selection !== "string" && selection.previousConfig
              ? (selection.previousConfig.nBatch ?? null)
              : useChatRuntimeStore.getState().nBatch;
          const previousNUbatch =
            typeof selection !== "string" && selection.previousConfig
              ? (selection.previousConfig.nUbatch ?? null)
              : useChatRuntimeStore.getState().nUbatch;
          // The outgoing tuning intent, read the same way and for the same
          // reason: a rollback restores the control, not the echo.
          const previousServerTuning: ServerTuningValues =
            typeof selection !== "string" && selection.previousConfig
              ? selection.previousConfig
              : useChatRuntimeStore.getState();
          // Same reason: the rollback echo would overwrite an edit staged against it.
          const previousMlxKvBits =
            typeof selection !== "string" && selection.previousConfig
              ? (selection.previousConfig.mlxKvBits ?? null)
              : useChatRuntimeStore.getState().mlxKvBits;
          if (isGguf && isDiffusion === undefined) {
            // Prepare the token exactly as validateModel/loadModel do (and as
            // the compare path does): the Hub rejects an invalid Authorization
            // header with 401 even for a public repo, so sending the raw stored
            // token here would abort the load before the existing "continue
            // anonymously / replace token" recovery flow could run.
            const preparedToken = await prepareHfTokenForUse(
              useChatRuntimeStore.getState().hfToken || null,
            );
            if (!preparedToken.proceed) {
              throw new Error("Model load cancelled.");
            }
            isDiffusion = (
              await fetchGgufStagedMetadata({
                model_path: loadPath,
                gguf_variant: ggufVariant ?? null,
                hf_token: preparedToken.token,
                nativePathToken: nativePathToken ?? null,
              })
            ).isDiffusion;
          }
          const targetIsDiffusion = isDiffusion === true;
          if (pendingLoadConfig) {
            applyPerModelConfigToRuntime(pendingLoadConfig, {
              isDiffusion: targetIsDiffusion,
            });
          }
          const currentCheckpoint =
            useChatRuntimeStore.getState().params.checkpoint;
          const stateBeforeUnload = useChatRuntimeStore.getState();
          const platform = usePlatformStore.getState();
          let trustRemoteCode = stateBeforeUnload.params.trustRemoteCode ?? false;
          let approvedRemoteCodeFingerprint: string | null = null;
          // A staged config carries its pin; one that applied the saved record and then
          // selected carries none, so the pre-move field is where it survives.
          const pinnedMaxSeqLength = normalizeMaxSeqLength(
            pendingLoadConfig
              ? pendingLoadConfig.maxSeqLength
              : resolveInitialConfig(modelId, ggufVariant).config.maxSeqLength,
          );
          const maxSeqLength =
            pinnedMaxSeqLength ?? stateBeforeUnload.params.maxSeqLength;
          const previousActiveNativePathToken =
            stateBeforeUnload.activeNativePathToken;
          const previousActiveLoadId = stateBeforeUnload.activeLoadId;
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
            (typeof selection !== "string"
              ? selection.previousConfig?.maxSeqLength
              : null) ?? maxSeqLength;
          // The intent the model had, not the length it ended up at: sending the
          // resolved length would pin a model nobody pinned.
          // The resident backend's own answer, so a native-audio checkpoint the worker
          // served off the MLX path does not roll back at the sentinel.
          const previousIsMlx = residentIsServedByMlx(
            previousIsGguf,
            platform.deviceType,
            platform.chatOnlyReason,
            stateBeforeUnload.loadedIsMlx,
          );
          // What the outgoing model loaded with, not the control's value: a pin typed
          // and never applied would change a window the failed switch never touched.
          const previousPin = stateBeforeUnload.loadedCustomContextLength;
          // It reloads at the pin it loaded with, whichever backend served it; only
          // llama.cpp's placement rules override that, where they own sizing.
          const rollbackMaxSeqLength = previousIsGguf
            ? resolveFitMaxSeqLength(
                previousIsGguf,
                stateBeforeUnload.loadedGpuMemoryMode ?? "auto",
                stateBeforeUnload.loadedGpuLayers ?? GPU_LAYERS_AUTO,
                stateBeforeUnload.loadedCustomContextLength,
                stateBeforeUnload.loadedContextLength ?? 0,
              )
            : (previousPin ??
              unpinnedLoadContext(false, previousIsMlx, previousMaxSeqLength));
          const hfToken = stateBeforeUnload.hfToken || null;
          const previousModelRequiresTrustRemoteCode =
            stateBeforeUnload.modelRequiresTrustRemoteCode;
          const previousActiveNativePathExpiresAtMs =
            stateBeforeUnload.activeNativePathExpiresAtMs;
          // Snapshot the load settings at click time, before the awaits below
          // (validation, the trust dialog, unload). When the picker staged a
          // config payload, prefer it over the store: React may not have
          // flushed NumericValueInput's blur commit into state yet.
          // Per-model: a template is written against one model's tokens.
          let loadChatTemplateOverride =
            pendingLoadConfig?.chatTemplateOverride?.trim()
              ? pendingLoadConfig.chatTemplateOverride
              : stateBeforeUnload.chatTemplateOverride;
          const loadKvCacheDtype =
            pendingLoadConfig?.kvCacheDtype ?? stateBeforeUnload.kvCacheDtype;
          // Per-model, not a standing preference: eligibility is decided per model.
          let loadMlxKvBits =
            pendingLoadConfig?.mlxKvBits ?? stateBeforeUnload.mlxKvBits;
          // gpuMemoryMode is a standing preference (kept across a model switch);
          // the rest are per-model knobs the reset below clears, so they are
          // re-baselined there in lock-step with the store.
          let loadCustomContextLength =
            pendingLoadConfig?.customContextLength ??
            stateBeforeUnload.customContextLength;
          const loadContextLength = stateBeforeUnload.loadedContextLength;
          const loadTensorParallel = targetIsDiffusion
            ? false
            : (pendingLoadConfig?.tensorParallel ??
              stateBeforeUnload.tensorParallel);
          // The diffusion runner has no projector to skip, so the toggle is inert
          // there for the same reason tensorParallel is.
          //
          // Unlike tensorParallel, this does NOT survive a model switch: it is
          // per-model config defaulting to vision on, so a target that saved none gets
          // that default, not the outgoing model's setting. Carrying it over loaded the
          // new model text-only and silently, because the dedupe comparison above
          // already builds its own view of an unconfigured switch from
          // DEFAULT_PER_MODEL_CONFIG. resetsPerModelSettings cannot repair it: this
          // constant is captured before that block runs.
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
          let loadSplitRatio = stateBeforeUnload.splitRatio;
          // Reconcile the persisted pick against the GPUs present now, so a stale
          // cross-host / now-hidden pick is dropped before /load rather than
          // rejected there. Warm the device cache first: load-on-selection can
          // run before any GPU hook mounted, and a cold cache would pass the
          // pick through unvalidated. validateGpuIds derives from this too.
          if (
            pendingLoadConfig?.selectedGpuIds !== undefined ||
            stateBeforeUnload.selectedGpuIds != null
          ) {
            await ensureGpuDeviceCache();
          }
          let loadSelectedGpuIds =
            pendingLoadConfig?.selectedGpuIds !== undefined
              ? reconcilePersistedGpuIds(
                  pendingLoadConfig.selectedGpuIds,
                  pendingLoadConfig.selectedGpuIndexKind,
                  targetIsDiffusion,
                )
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
          // No store fallback and no reset with the rest below: undefined means
          // "this config never read them", and the route preserves the stored
          // flags when the field is omitted. Falling back to another model's
          // value, or to null, would clear flags the user set elsewhere.
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
            // Lightweight pre-flight validation: avoid unloading a working model
            // if the new identifier is clearly invalid (e.g. bad HF id / path).
            const validateNativePathLease = nativePathToken
              ? (await consumeNativePathToken(nativePathToken, "validate-model")).nativePathLease
              : undefined;
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
            // Per-model: the reset re-baselines to the staged config, like the load.
            const validateNParallel = resetsPerModelSettings
              ? (pendingLoadConfig?.nParallel ?? null)
              : loadNParallel;
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
              nativePathLease: validateNativePathLease,
              hf_token: hfToken,
              max_seq_length: validateMaxSeqLength,
              load_in_4bit: true,
              is_lora: isLora,
              gguf_variant: ggufVariant ?? null,
              cache_type_kv: loadKvCacheDtype,
              tensor_parallel: loadTensorParallel,
              disable_vision: loadDisableVision,
              gpu_ids: validateGpuIds ?? undefined,
              ...(isGguf
                ? {
                    gpu_memory_mode: loadGpuMemoryMode,
                    // Sized like the follow-up /load: else a manual DiffusionGemma
                    // split 409s during training even when it fits.
                    gpu_layers: validateGpuLayers,
                    n_parallel: validateNParallel,
                    // omitted when blank, like the load payload below
                    ...(validateNBatch != null
                      ? { n_batch: validateNBatch }
                      : {}),
                    ...(validateNUbatch != null
                      ? { n_ubatch: validateNUbatch }
                      : {}),
                    // Same omit-when-blank rule, and the same values: the preflight
                    // has to approve the command the follow-up /load sends.
                    ...serverTuningLoadPayload(validateServerTuning),
                    // The same list the load below sends. A --ctx-size or cache
                    // override in here changes the memory this preflight estimates,
                    // so omitting it approves a different command: during training
                    // that means approving the switch, unloading the resident model,
                    // and having /load refuse the target with the real arguments.
                    ...(!targetIsDiffusion && loadLlamaExtraArgs !== undefined
                      ? { llama_extra_args: loadLlamaExtraArgs ?? [] }
                      : {}),
                  }
                : {}),
            });
            // Upgrade consent runs before the security dialogs; Accept installs and the load continues.
            if (validation.requires_transformers_upgrade) {
              const upgraded = await confirmTransformersUpgradeIfNeeded({
                modelName: modelId,
                upgrade: validation.transformers_upgrade,
                // No installable release: custom-code models may fall back to the trust_remote_code gate below.
                trustRemoteCodeFallback: validation.requires_trust_remote_code,
                // The install refuses while chats generate and takes no force flag of its own, so
                // without this the "Stop and reload" the user just confirmed dies here: Retry hits
                // the same 409, and this path leaves chats running.
                forceCancelActive,
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
              });
              if (!approved) {
                throw new Error(getTrustRemoteCodeRequiredMessage(displayName));
              }
            }
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");
            const loadNativePathLease = nativePathToken
              ? (await consumeNativePathToken(nativePathToken, "load-model")).nativePathLease
              : undefined;

            cancelPreStreamRunReservations(stopDecision.preStreamRunTokens);
            requestLocalPromptQueueStop(stopDecision.promptQueueThreadIds);
            if (currentCheckpoint) {
              // With chats generating, skip this preliminary unload: it cancels them ahead of /load's
              // preflight, so a rejected target truncates replies for a model that never loads
              // (/load evicts past those checks itself). Idle, unload first and free VRAM early.
              if (!forceCancelActive) {
                await unloadModel({ model_path: currentCheckpoint });
              }
              // Set either way: /load can still leave no model resident, and an unneeded rollback
              // hits already_loaded before the gate.
              previousWasUnloaded = true;
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
                // Per-model too: a different model follows the server default
                // unless its staged config overrides it.
                nParallel: null,
                loadedNParallel: null,
                nBatch: null,
                loadedNBatch: null,
                nUbatch: null,
                loadedNUbatch: null,
                // Per-model too, and cleared in both halves: a baseline left from
                // the model that just went would be re-sent by a later rollback.
                ...clearedServerTuningState(),
                // Per-model GPU knobs must not follow onto a different model
                // (gpuMemoryMode is a standing preference and is kept).
                selectedGpuIds: null,
                selectedGpuIndexKind: null,
                gpuLayers: GPU_LAYERS_AUTO,
                nCpuMoe: 0,
                splitRatio: null,
                // A Manual+Auto context pin is per-model; clear it so a different
                // model loads at Auto/native, not the previous model's pin.
                customContextLength: null,
              });
              loadSpeculativeType =
                pendingLoadConfig?.speculativeType != null
                  ? normalizeSpeculativeType(pendingLoadConfig.speculativeType)
                  : persistedSpeculativeType;
              loadSpecDraftNMax = pendingLoadConfig?.specDraftNMax ?? null;
              loadNParallel = pendingLoadConfig?.nParallel ?? null;
              loadNBatch = pendingLoadConfig?.nBatch ?? null;
              loadNUbatch = pendingLoadConfig?.nUbatch ?? null;
              loadServerTuning = {
                loadMode: pendingLoadConfig?.loadMode ?? null,
                specDraftCacheDtype:
                  pendingLoadConfig?.specDraftCacheDtype ?? null,
                ctxCheckpoints: pendingLoadConfig?.ctxCheckpoints ?? null,
                cacheRam: pendingLoadConfig?.cacheRam ?? null,
              };
              // Both payload-only. The store keeps its values: a width is dormant
              // preset state off MLX, and a completed load rewrites both anyway.
              loadMlxKvBits = pendingLoadConfig?.mlxKvBits ?? null;
              loadChatTemplateOverride =
                pendingLoadConfig?.chatTemplateOverride?.trim()
                  ? pendingLoadConfig.chatTemplateOverride
                  : null;
              // Keep the click-time snapshot in lock-step with the store reset so
              // the load below sizes against the cleared per-model knobs, not the
              // previous model's (gpuMemoryMode is standing, so left as captured).
              // An explicit staged config from run-settings still wins.
              loadCustomContextLength =
                pendingLoadConfig?.customContextLength ?? null;
              loadSelectedGpuIds =
                pendingLoadConfig?.selectedGpuIds !== undefined
                  ? reconcilePersistedGpuIds(
                      pendingLoadConfig.selectedGpuIds,
                      pendingLoadConfig.selectedGpuIndexKind,
                      targetIsDiffusion,
                    )
                  : null;
              loadGpuLayers = pendingLoadConfig?.gpuLayers ?? GPU_LAYERS_AUTO;
              loadNCpuMoe = pendingLoadConfig?.nCpuMoe ?? 0;
              loadSplitRatio = null;
            }

            // The Context Length the USER set for this load, captured before the
            // clamp below can stand in for it. This, not the n_ctx that goes on
            // the wire, is what the completed load pins: several send rules put a
            // positive n_ctx on the wire with the control on Auto, and a pin read
            // back out of one of those is a number the user never chose.
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
            // Pinning layers on the SAME model keeps the currently resolved
            // context: with no explicit pin, a manual+pinned reload would send 0,
            // which the backend's --fit off branch treats as the NATIVE context --
            // far larger than the sheet shows when the load was fit-sized (Default
            // or Manual + Auto layers may auto-reduce context to fit VRAM), a
            // likely OOM. loadedContextLength is that resolved value; a model already
            // at native reloads unchanged, so this is safe for any prior mode.
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
            // A queue can be created while the preliminary unload is pending.
            // Stop a second time at the final boundary so no prompt captured
            // against the outgoing checkpoint survives into the new backend.
            requestLocalPromptQueueStop();
            const loadResponse = await loadModel({
              model_path: loadPath,
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
              mlx_kv_bits: loadMlxKvBits ?? null,
              speculative_type: loadSpeculativeType,
              spec_draft_n_max: loadSpecDraftNMax,
              // GGUF-only: slots mean nothing for a transformers load.
              n_parallel: isGguf ? loadNParallel : null,
              // Sent only once known, and [] is the explicit "launch with none":
              // the flags are llama-server's, so a transformers load never carries
              // them, and neither does a diffusion GGUF: that one is GGUF-shaped but
              // runs through the visual runner, which builds its command without
              // these, so sending them would record arguments the process never got.
              ...(isGguf && !targetIsDiffusion && loadLlamaExtraArgs !== undefined
                ? { llama_extra_args: loadLlamaExtraArgs ?? [] }
                : {}),
              // omitted when blank: a null counts as set and strips inherited -b / -ub
              ...(isGguf && loadNBatch != null ? { n_batch: loadNBatch } : {}),
              ...(isGguf && loadNUbatch != null
                ? { n_ubatch: loadNUbatch }
                : {}),
              // llama-server's own, so gated like the extras above: GGUF, and not
              // the diffusion runner, which builds its command without them.
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
            });
            cpuFallbackReason = loadResponse.cpu_fallback_reason ?? null;
            mmprojFallbackReason = loadResponse.mmproj_fallback_reason ?? null;

            // If cancelled while loading, don't update UI to show
            // the model as active -- it's being unloaded.
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");

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
            const loadedFields = loadedContextFields(loadResponse);
            // The reported window, or where nothing sized one the requested length.
            // The request answers only for a backend that sizes nothing: a self-sizing
            // one was sent the sentinel, which as a budget is zero.
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
                // The served window, as background and compare loads already record,
                // or the active model reports a context it is not running at.
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
              // Lay this model's remembered settings back over its defaults,
              // but not a budget larger than the context it just loaded with.
              {
                fromModelDefaults: true,
                maxTokensCap: loadedContextCap,
              },
            );
            // Qwen3.5/3.6 small models (0.8B, 2B, 4B, 9B) disable thinking by default.
            // Anchored regex: first "Xb" / "X.Xb" after start-of-string or
            // [-_/.] so the version literal in "qwen3.5" / "qwen3.6" doesn't
            // match first, and for "Qwen3.5-35B-A3B" the result is 35 (total
            // params), not 3 (MoE active params).
            let reasoningDefault = loadResponse.supports_reasoning ?? false;
            if (reasoningDefault) {
              const mid = modelId.toLowerCase();
              if (mid.includes("qwen3.5") || mid.includes("qwen3.6")) {
                // Scan path segments right to left so the size nearest the leaf
                // wins over a size-like parent dir; trailing boundary stops "8bit".
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
            // Slots the load actually committed. Non-GGUF never sends them and
            // diffusion ignores --parallel, so a click-time count on either
            // would mint a phantom override a saved preset carries onto a GGUF.
            const committedSlots =
              (loadResponse.is_gguf ?? false) &&
              !(loadResponse.is_diffusion ?? false)
                ? (loadNParallel ?? null)
                : null;
            // same rule for the batch sizes: gguf-only llama-server flags
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
            // The user's own Context Length (see explicitCtxPin), so an Auto load
            // stays Auto whatever n_ctx the send rules resolved for it. MLX pins the
            // same way, so it is admitted here rather than cleared as a non-GGUF.
            const keepCustomCtx = resolveExplicitCtxPin(
              loadResponse.is_gguf || targetIsMlx ? explicitCtxPin : null,
            );
            const reasoningAlwaysOn = loadResponse.reasoning_always_on ?? false;
            const reasoningStyle = loadResponse.reasoning_style ?? "enable_thinking";
            const supportsReasoning = loadResponse.supports_reasoning ?? false;
            const supportsPreserveThinking =
              loadResponse.supports_preserve_thinking ?? false;
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
              // Repaired from the echo like the knob above: loadDisableVision
              // forces the flag off for a diffusion target without writing the
              // store, so a Vision-off GGUF followed by a diffusion load would
              // otherwise leave the switch off over a load that never sent it.
              disableVision: loadResponse.disable_vision ?? false,
              // Set alongside loadedIsMultimodal so the composer can say WHY
              // images are unavailable.
              loadedVisionDisabledByUser:
                loadResponse.vision_disabled_by_user ?? false,
              ...loadedGpuMemoryFields(loadResponse),
              speculativeType: loadedSpec,
              loadedSpeculativeType: loadedSpec,
              specDraftNMax: loadResponse.spec_draft_n_max ?? null,
              loadedSpecDraftNMax: loadResponse.spec_draft_n_max ?? null,
              // Keep the click-time value: the echo is the resolved count, and
              // adopting it would pin a blank "server default" control.
              nParallel: committedSlots,
              loadedNParallel: committedSlots,
              nBatch: committedNBatch,
              loadedNBatch: committedNBatch,
              ...committedServerTuning,
              // What this model is running, for the rollback below. An omitted field
              // inherited the resident process's list, so the last thing we knew
              // still holds unless this was a different model.
              //
              // An explicit empty list is recorded as empty, not as null: the
              // rollback sends this field only when it has one, and omitting it is
              // what makes /load inherit, so a model that was launched with no
              // extras would come back carrying the arguments of the load that just
              // failed. null stays for "we were never told".
              //
              // The server's own echo first, since it is the only account of what
              // the launch actually carried: a reload that omits the field but sets
              // max_seq_length has its inherited --ctx-size stripped before launch,
              // and the status refresh that would notice runs while modelLoading is
              // still true and reseeds nothing. Without this the next rollback
              // resent a flag the successful reload had removed.
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
            // Unlock attach menus for capabilities the catalog entry lacked.
            syncModelCapabilities(modelId, loadResponse);
            // Qwen3-family: apply thinking-mode-specific params after load.
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
                // Same rule as the load response: defaults first, this model's
                // remembered settings over them.
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
            // Remembered so auto-load re-picks what the user ran, not the
            // smallest. Native file-picker paths need a signed, expiring lease,
            // so they stay out.
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
                const rollbackResponse = await loadModel({
                  // The pin it loaded from: without it this retries the ref that needed pinning.
                  model_path: previousActiveLoadId || previousCheckpoint,
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
                  mlx_kv_bits: stateBeforeUnload.loadedMlxKvBitsRequested,
                  speculative_type:
                    stateBeforeUnload.loadedSpeculativeType,
                  spec_draft_n_max:
                    stateBeforeUnload.loadedSpecDraftNMax,
                  n_parallel: stateBeforeUnload.loadedNParallel,
                  // omit unset fields: a null counts as set and would strip the previous server's extras
                  ...(stateBeforeUnload.loadedNBatch != null
                    ? { n_batch: stateBeforeUnload.loadedNBatch }
                    : {}),
                  ...(stateBeforeUnload.loadedNUbatch != null
                    ? { n_ubatch: stateBeforeUnload.loadedNUbatch }
                    : {}),
                  ...serverTuningLoadPayload({
                    loadMode: stateBeforeUnload.loadedLoadMode,
                    specDraftCacheDtype:
                      stateBeforeUnload.loadedSpecDraftCacheDtype,
                    ctxCheckpoints: stateBeforeUnload.loadedCtxCheckpoints,
                    cacheRam: stateBeforeUnload.loadedCacheRam,
                  }),
                  // Explicit, unlike the batch pair above: the failed switch left the
                  // TARGET resident, so an omitted field here inherits across models,
                  // which the route refuses, and the previous model would come back
                  // without the arguments it was running.
                  ...(stateBeforeUnload.loadedLlamaExtraArgs != null
                    ? { llama_extra_args: stateBeforeUnload.loadedLlamaExtraArgs }
                    : {}),
                  // Restore the previous model in the split mode it was running,
                  // not the default layer split.
                  tensor_parallel: stateBeforeUnload.loadedTensorParallel ?? false,
                  // What the PREVIOUS server was loaded with. Not the control field:
                  // applyPerModelConfigToRuntime writes disableVision before this
                  // baseline is captured, so it holds the TARGET's setting. Not
                  // loadedVisionDisabledByUser either: that is narrowed to models that
                  // can do images, so a non-vision GGUF would come back with vision on
                  // while the control restored to off. Same separately tracked baseline
                  // tensor_parallel uses above.
                  disable_vision: stateBeforeUnload.loadedDisableVision ?? false,
                  // Restore the previous model's GPU Memory placement, not backend defaults.
                  gpu_memory_mode: stateBeforeUnload.loadedGpuMemoryMode ?? "auto",
                  gpu_layers: stateBeforeUnload.loadedGpuLayers ?? GPU_LAYERS_AUTO,
                  // A recovered Vulkan model needs its staged CPU-only runtime back
                  // after the failed target load unloaded the live server.
                  cpu_fallback: stateBeforeUnload.loadedCpuFallback,
                  n_cpu_moe: stateBeforeUnload.loadedNCpuMoe ?? 0,
                  tensor_split: stateBeforeUnload.loadedSplitRatio ?? undefined,
                  gpu_ids: stateBeforeUnload.loadedGpuIds ?? undefined,
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
                  // Restore the previous token's lease together with the token so a
                  // rollback never pairs restored token A with failed load B's expiry.
                  activeNativePathExpiresAtMs: previousActiveNativePathToken
                    ? (previousActiveNativePathExpiresAtMs ?? null)
                    : null,
                  // Restore the editable speculative knobs to the rolled-back
                  // model's; the loaded baselines below come from its reload echo.
                  speculativeType: stateBeforeUnload.loadedSpeculativeType ?? null,
                  specDraftNMax: stateBeforeUnload.loadedSpecDraftNMax ?? null,
                  // Control keeps its intent; only the baseline takes the echo.
                  nParallel: previousNParallel,
                  loadedNParallel: stateBeforeUnload.loadedNParallel ?? null,
                  nBatch: previousNBatch,
                  loadedNBatch: stateBeforeUnload.loadedNBatch ?? null,
                  nUbatch: previousNUbatch,
                  loadedNUbatch: stateBeforeUnload.loadedNUbatch ?? null,
                  // Same split: the controls keep the outgoing model's intent and
                  // the baselines come from what that model was launched with.
                  loadMode: previousServerTuning.loadMode ?? null,
                  loadedLoadMode: stateBeforeUnload.loadedLoadMode ?? null,
                  specDraftCacheDtype:
                    previousServerTuning.specDraftCacheDtype ?? null,
                  loadedSpecDraftCacheDtype:
                    stateBeforeUnload.loadedSpecDraftCacheDtype ?? null,
                  ctxCheckpoints: previousServerTuning.ctxCheckpoints ?? null,
                  loadedCtxCheckpoints:
                    stateBeforeUnload.loadedCtxCheckpoints ?? null,
                  cacheRam: previousServerTuning.cacheRam ?? null,
                  loadedCacheRam: stateBeforeUnload.loadedCacheRam ?? null,
                  loadedSpeculativeType: rollbackSpeculativeType,
                  loadedSpecDraftNMax:
                    rollbackResponse.spec_draft_n_max ?? null,
                  loadedKvCacheDtype: rollbackResponse.cache_type_kv ?? null,
                  ...mlxRuntimeStateFrom(rollbackResponse),
                  // After the spread, which seeds the control from the echo; the
                  // control keeps its intent, like nParallel above.
                  mlxKvBits: previousMlxKvBits,
                  loadedChatTemplateOverride:
                    stateBeforeUnload.loadedChatTemplateOverride,
                  ...loadedGpuMemoryFields(rollbackResponse),
                  tensorParallel: rollbackResponse.tensor_parallel ?? false,
                  loadedTensorParallel:
                    rollbackResponse.tensor_parallel ?? false,
                  loadedDisableVision:
                    rollbackResponse.disable_vision ?? false,
                  // The rolled-back model's own loaded value, matching the request
                  // above field for field. Not stateBeforeUnload.disableVision, which
                  // holds the TARGET's value by now and would show Vision off over a
                  // restored projector, arming the next Apply to switch it off for real.
                  // Not the echo either: the replayed request is gated on the model
                  // having a projector, so a text-only GGUF switched off echoes false
                  // and would flip Vision back on.
                  disableVision: stateBeforeUnload.loadedDisableVision ?? false,
                  loadedVisionDisabledByUser:
                    rollbackResponse.vision_disabled_by_user ?? false,
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

        // Poll download progress for non-cached models, then (after download
        // or for cached models) poll the llama-server mmap phase so "Starting
        // model..." doesn't look frozen for minutes on large MoE models.
        let progressInterval: ReturnType<typeof setInterval> | null = null;
        const expectedBytes =
          typeof selection !== "string" ? selection.expectedBytes ?? 0 : 0;

        // One buffer per phase, so a flip cannot price the new phase against
        // the old one's clock. Was a private copy of the shared estimator, so
        // fixes never reached this toast. The helper takes SECONDS, not ms.
        const dlSamples: TransferSample[] = [];
        const mmapSamples: TransferSample[] = [];

        function estimate(
          samples: TransferSample[],
          bytes: number,
          total: number,
        ): { rate: number; eta: number; stable: boolean } {
          if (typeof document !== "undefined" && document.hidden) {
            // This 2s interval is clamped to about once a minute while hidden,
            // and the estimator reads gaps as the burst cadence. The hub poll
            // loop and the voice poller drop them the same way.
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

        let downloadComplete = isDownloaded || isCachedLora;

        const pollDownload = async () => {
          if (abortCtrl.signal.aborted || !loadingModelRef.current) {
            if (progressInterval) clearInterval(progressInterval);
            return;
          }
          try {
            const prog =
              ggufVariant && expectedBytes > 0
                ? await getGgufDownloadProgress(modelId, ggufVariant, expectedBytes)
                : await getDownloadProgress(modelId);
            if (!loadingModelRef.current) return;

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
              // loadProgress state is only read by the dismissed-toast inline
              // status. Writing it while the toast is visible re-renders the
              // whole chat page every poll — cheap in Chrome, janky in the
              // desktop WebView2 (laggy typing). Feed the toast directly and
              // only touch state when the inline view is actually live.
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
              // Inline-status-only state; skip the chat-page re-render unless it's shown.
              if (loadToastDismissedRef.current) {
                setLoadProgress({
                  percent: null,
                  label: `${dlGb.toFixed(1)} GB downloaded${rateSuffix}`,
                  phase: "downloading",
                });
              }
            } else if (prog.progress >= 1 && hasShownProgress) {
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
                      "Download complete. Loading the model into memory.",
                      100,
                      "Download complete",
                    ),
                  ),
                });
              }
              // Keep polling: the mmap branch below takes over from here.
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
            // Inline-status-only state (see pollDownload): while the toast is
            // up, skip the state write so the chat page doesn't re-render every
            // poll during "Starting model" — the desktop WebView2 typing-lag fix.
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
          // Same composition as the auto-load path, through the same helper, so the
          // two cannot describe an identical failure differently again.
          const notice = loadFallbackNotice(
            `${toastDisplayName} loaded`,
            cpuFallbackReason,
            mmprojFallbackReason,
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
          }
          throw err;
        } finally {
          if (progressInterval) clearInterval(progressInterval);
          resetLoadingUi();
          if (postLoadRefresh.needed && !abortCtrl.signal.aborted) {
            void refreshContextUsage({ afterModelLoad: true });
          }
        }
      } catch (error) {
        restorePreviousConfig();
        if (abortCtrl.signal.aborted) return; // User cancelled, nothing to report
        resetLoadingUi();
        const message =
          error instanceof Error ? error.message : "Failed to load model";
        setModelsError(message);
        setLastModelLoadError(message); // load-specific failure for the attach gates
        if (throwOnError) {
          throw error instanceof Error ? error : new Error(message);
        }
      }
    },
    [
      cancelLoading,
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
    const bailIfLoading = (): boolean => {
      const runtime = useChatRuntimeStore.getState();
      if (!runtime.modelLoading && !runtime.loadingModelPick) return false;
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
      // Block queue materialization before taking the confirmation snapshot.
      // Otherwise a queue can appear while the dialog is open and be stopped
      // by the unload even though the user never confirmed stopping it.
      lifecycleLease = useChatRuntimeStore.getState().beginModelLoading();
      if (lifecycleLease === null) {
        return false;
      }
      // Ejecting tears down llama-server, so every chat stops. Same prompt, but it
      // leaves no model loaded, so it must not be worded as a reload.
      const stopDecision = await confirmStopRunningChatsIfNeeded(
        "Unloading the model",
        "unload",
      );
      if (!stopDecision.proceed) return false;

      async function performUnload(): Promise<void> {
        cancelPreStreamRunReservations(stopDecision.preStreamRunTokens);
        requestLocalPromptQueueStop(stopDecision.promptQueueThreadIds);
        await unloadModel({
          model_path: params.checkpoint,
          force_cancel_active: stopDecision.forceCancelActive,
        });
        requestLocalPromptQueueStop();
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
    } finally {
      if (lifecycleLease !== null) {
        useChatRuntimeStore.getState().endModelLoading(lifecycleLease);
      }
    }
  }, [clearCheckpoint, params.checkpoint, refresh, setModelsError]);

  return {
    refresh,
    selectModel,
    ejectModel,
    cancelLoading,
    loadingModel,
    loadProgress,
    loadToastDismissed,
  };
}
