# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A failed auto-load of a cached model must not become a Hub download.

Runs the real ``autoLoadSmallestModel`` from chat-adapter.ts under node with the module boundary
stubbed, so these assert behaviour, not source text. The sweep's catches are parameterless, so a
cached repo whose load rejected fell through to fetching an unrelated default model.
"""

import json
import os
import shutil
import subprocess
import re
import tempfile
import textwrap
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]


def _source_path(relative_path: str) -> Path:
    direct = WORKDIR / relative_path
    if direct.exists():
        return direct
    return WORKDIR / "unsloth_repo" / relative_path


ADAPTER = _source_path("studio/frontend/src/features/chat/api/chat-adapter.ts")
# Inlined for real, not stubbed: waitForSettledServerStatus raises this gate so an ordinary
# refresh cannot publish a mid-replacement status as the pick underneath it.
WAIT_GATE = _source_path("studio/frontend/src/features/chat/lib/server-model-wait.ts")


def _wait_gate_source() -> str:
    """The gate module with its two ponyfill imports dropped; PONYFILLS supplies those."""
    gate = "\n".join(
        line
        for line in WAIT_GATE.read_text(encoding = "utf-8").splitlines()
        if not line.startswith(
            (
                "import ",
                "  disposableTimeoutSignal,",
                "  pollSignal,",
                "  type PollSignal,",
                '} from "@/features/hub',
            )
        )
    ).replace(": PollSignal", "")
    assert "export function beginServerModelWait(" in gate
    return gate


# Short so no scenario waits out the real 30s cap; the shipped value is asserted in
# tests/studio/test_chat_mount_cli_load_adoption.py.
PONYFILLS = """
export function disposableTimeoutSignal(_ms: number) {
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(new DOMException("The operation timed out.", "TimeoutError")),
    1200,
  );
  return { signal: controller.signal, dispose: () => clearTimeout(timer) };
}
export function pollSignal(parent: AbortSignal, ms: number) {
  const timeout = disposableTimeoutSignal(ms);
  const controller = new AbortController();
  const abort = (reason: unknown) => {
    if (!controller.signal.aborted) controller.abort(reason);
  };
  const onParent = () => abort(parent.reason);
  const onTimeout = () => abort(timeout.signal.reason);
  parent.addEventListener("abort", onParent, { once: true });
  timeout.signal.addEventListener("abort", onTimeout, { once: true });
  if (parent.aborted) onParent();
  return {
    signal: controller.signal,
    dispose: () => {
      parent.removeEventListener("abort", onParent);
      timeout.signal.removeEventListener("abort", onTimeout);
      timeout.dispose();
    },
  };
}
"""
TEMP = WORKDIR / "temp" / "chat_autoload_failure_gate"
DEFAULT_MODEL = "unsloth/gemma-4-E2B-it-GGUF"
GEMMA_REPO = "unsloth/gemma-4-26B-A4B-it-qat-GGUF"
LOCAL_GGUF_PATH = "/home/john-doe/models/qwen3-1.7b-instruct-q4_k_m.gguf"

# Stubs for everything autoLoadSmallestModel imports;
# each scenario supplies the cache inventory and how /validate and /load answer per model_path.
PREAMBLE = """
type LastLocalModelKind = "gguf" | "model";
type GgufVariantDetail = {
  quant?: string | null;
  filename?: string | null;
  downloaded?: boolean;
  size_bytes: number;
};
type ChatModelSummary = Record<string, unknown>;

export type Scenario = {
  ggufRepos: any[];
  modelRepos: any[];
  localModels: any[];
  chatOnly: boolean;
  deviceType: string;
  /** Click the toast's Cancel while requestStart is still in its preflights. */
  cancelDuringStart: boolean;
  /** Cancellation request fails and the transfer keeps running. */
  cancelFails: boolean;
  /** "decline" or "anonymous" from the expired-token dialog. */
  tokenDecision: string | null;
  /** false = the backend platform probe has not landed yet. */
  platformFetched: boolean;
  variants: Record<string, any>;
  lastLoaded: any;
  validate: (payload: any) => any;
  load: (payload: any) => any;
  /** How the Hub download manager answers the default-model request:
   *  "complete" (default), "cancelled", "failed", "busy", or "conflict". */
  download: (request: any) => string;
  // Which backend the host serves with, and what the record holds: both decide the ask.
  platform?: { deviceType: string; chatOnlyReason: string | null };
  config?: Record<string, unknown>;
};

// The stored-arguments hydration the auto-load runs before it calls /load. Neutral
// here on purpose: these scenarios are about the failure gate, not about which flags
// a model launches with, and a stub that invented some would change the /load payload
// every scenario asserts on. fetchLoadExtraArgs answers "nothing stored", which is
// what a fresh install has, and the sanitizer is identity so a scenario that does set
// llamaExtraArgs still sends exactly what it set.
export async function loadManagedLlamaFlags(): Promise<any> {
  return null;
}
export async function fetchLoadExtraArgs(
  _loadId: string,
  _aliasId?: string | null,
  _variant?: string | null,
): Promise<{ tokens: string[]; explicit: boolean }> {
  return { tokens: [], explicit: false };
}
export function sanitizeStoredExtraArgs(
  tokens: readonly string[],
  _managed: ReadonlySet<string>,
  _limits?: any,
): string[] {
  return [...tokens];
}

// The llama-server tuning group, mirroring
// studio/frontend/src/features/chat/lib/server-tuning-fields.ts. Real logic rather
// than a neutral stub: serverTuningLoadPayload spreads into the /load payload these
// scenarios assert on, so a stub that always returned {} would keep passing if the
// real one started sending a field. These are pure and import nothing, so copying
// them costs only the drift this file already guards against by name.
export function serverTuningLoadPayload(values: any): any {
  return {
    ...(values?.loadMode != null ? { load_mode: values.loadMode } : {}),
    ...(values?.specDraftCacheDtype != null
      ? { spec_draft_cache_type: values.specDraftCacheDtype }
      : {}),
    ...(values?.ctxCheckpoints != null
      ? { ctx_checkpoints: values.ctxCheckpoints }
      : {}),
    ...(values?.cacheRam != null ? { cache_ram: values.cacheRam } : {}),
  };
}
export function clearedServerTuningState(): any {
  return {
    loadMode: null,
    loadedLoadMode: null,
    specDraftCacheDtype: null,
    loadedSpecDraftCacheDtype: null,
    ctxCheckpoints: null,
    loadedCtxCheckpoints: null,
    cacheRam: null,
    loadedCacheRam: null,
  };
}
export function committedServerTuningState(values: any, isDiffusion = false): any {
  if (isDiffusion) {
    return clearedServerTuningState();
  }
  const loadMode = values?.loadMode ?? null;
  const specDraftCacheDtype = values?.specDraftCacheDtype ?? null;
  const ctxCheckpoints = values?.ctxCheckpoints ?? null;
  const cacheRam = values?.cacheRam ?? null;
  return {
    loadMode,
    loadedLoadMode: loadMode,
    specDraftCacheDtype,
    loadedSpecDraftCacheDtype: specDraftCacheDtype,
    ctxCheckpoints,
    loadedCtxCheckpoints: ctxCheckpoints,
    cacheRam,
    loadedCacheRam: cacheRam,
  };
}

export const EVENTS: any[] = [];
let SCENARIO: Scenario;
export function setScenario(scenario: Scenario) {
  SCENARIO = scenario;
  EVENTS.length = 0;
  STORE = makeStore();
  DOWNLOAD_LISTENERS = [];
  DOWNLOAD_STATE = { jobs: {} };
  DOWNLOAD_SUBS = [];
  CANCELLED_KEYS = new Set();
  CANCEL_TOAST_ACTION = null;
}

const GPU_LAYERS_AUTO = -1;

// The sliced region now includes the queue-specific empty-model resolver and
// visible-state snapshot helpers, so their imported contracts must exist even
// though these scenarios call autoLoadSmallestModel directly.
let STATUS_CALLS = 0;
async function getInferenceStatus() {
  const call = STATUS_CALLS++;
  if (call < (SCENARIO.statusFailures ?? 0)) {
    throw new Error("status unavailable");
  }
  const loading = SCENARIO.serverLoading ?? [];
  const clearsAfter = SCENARIO.serverLoadingClearsAfter ?? null;
  const settled = clearsAfter !== null && call >= clearsAfter;
  // A replacement keeps the outgoing model resident and reported until the
  // incoming one lands, so status names both at once.
  const resident = SCENARIO.serverResident ?? null;
  const active = settled ? (loading[0] ?? resident) : resident;
  return {
    active_model: active,
    model_identifier: active,
    loading: settled ? [] : loading,
  };
}
// Defined above the sliced region in chat-adapter.ts, so the slice would call a
// name the harness lacks. Reached only when the lease is already held, which no
// scenario here does.
async function waitForModelReady(_signal?: any) {}
const window: any = { location: { href: "http://localhost/chat" } };
function isExternalModelId(value: unknown) {
  return typeof value === "string" && value.startsWith("external::");
}
// chat-runtime-store.ts:
//   return storedPreserveThinking ?? preserveThinkingDefaultFromLoad(resp);
// No scenario here sets a stored preference, so the stub is the model-family
// default the backend resolves, verbatim from resolve-preserve-thinking-default.ts:
//   Boolean(resp.supports_preserve_thinking && resp.preserve_thinking_default)
// Present because the sliced region calls it; without it every scenario in this
// file fails on the harness guard rather than on anything it means to test.
function resolvePreserveThinkingOnLoad(resp: any) {
  return Boolean(resp?.supports_preserve_thinking && resp?.preserve_thinking_default);
}

function resolveInferenceCheckpointId(status: any) {
  return status.active_model
    ? (status.model_identifier ?? status.active_model)
    : null;
}
function snapshotQueuedChatRunSettings(state: any) {
  return { ...state, params: { ...state.params } };
}
// Mirrored rather than stubbed to constants, so a scenario that pins is not auto-answered.
function loadedContextFields(resp: any) {
  if (!resp) {
    return {
      loadedContextLength: null, maxContextLength: null,
      nativeContextLength: null, loadedIsGguf: null,
      loadedContextEnforced: null,
    };
  }
  const isGguf = resp.is_gguf ?? false;
  const loaded = resp.context_length ?? null;
  // !is_mlx as well: MLX sizes its own window, so it reports one without a native.
  if (!isGguf && !resp.is_mlx && resp.native_context_length == null) {
    return {
      loadedContextLength: null, maxContextLength: null,
      nativeContextLength: null, loadedIsGguf: false,
      loadedContextEnforced: null,
    };
  }
  return {
    loadedContextLength: loaded,
    maxContextLength: resp.max_context_length ?? loaded,
    nativeContextLength: resp.native_context_length ?? null,
    loadedIsGguf: isGguf,
    loadedContextEnforced: isGguf ? true : (resp.context_length_enforced ?? null),
  };
}
const NO_MLX_REASONS = new Set(["mlx_unavailable", "intel_mac", "detection_failed"]);
function isServedByMlx(isGguf: boolean, deviceType: string, reason: unknown) {
  return !isGguf && deviceType === "mac" && !NO_MLX_REASONS.has((reason ?? "") as string);
}
function retainedContextPin(args: any) {
  if (args.isGguf) {
    return args.gpuMemoryMode === "manual" && args.gpuLayers < 0
      ? ((args.requestedContextLength ?? 0) > 0 ? args.requestedContextLength : null)
      : null;
  }
  if (args.isMlx) {
    return (args.requestedContextLength ?? 0) > 0 ? args.requestedContextLength : null;
  }
  return null;
}

function makeStore(): any {
  const state: any = {
    hfToken: null,
    // The visible selection the queued resolver reads; external means the turn
    // has to resolve a local model of its own.
    params: { maxSeqLength: 4096, checkpoint: SCENARIO?.visibleCheckpoint ?? "" },
    activeGgufVariant: null,
    activePresetSource: null,
    gpuMemoryMode: "auto",
    selectedGpuIds: null,
    models: [],
    modelLoading: false,
    activeThreadEpoch: 0,
    queuedSettingsEpoch: 0,
    contextUsage: null,
    contextUsageByThreadId: {},
    setCheckpoint: () => {},
    setModelRequiresTrustRemoteCode: () => {},
    setParams: (p: any) => { state.params = p; },
    setModels: (m: any[]) => { state.models = m; },
    beginModelLoading: () => {
      if (state.modelLoading) return null;
      state.modelLoading = true;
      return { id: 1 };
    },
    endModelLoading: () => { state.modelLoading = false; },
  };
  return state;
}
let STORE: any = makeStore();

// Imported by the sliced region from ../hooks/use-chat-model-runtime, so the
// harness has to supply it or the call is a bare ReferenceError -- which the
// retry loop below catches and scores as a failed load, making a healthy model
// look broken. Mirrors the real upsert closely enough for these scenarios: the
// tests assert on which models are loaded, not on the capability fields.
// Both sit behind `candidate.kind === "gguf" && (selectedGpuIds != null ||
// tensorParallel)`, which no scenario here sets, so they are unreached today --
// but they are the same landmine as syncModelCapabilities was, waiting for the
// first scenario that turns a GPU option on. Stubbed so the harness is complete
// rather than complete-by-luck.
// Imported by chat-adapter.ts from ../lib/mlx-runtime-state. Mirrors the real
// one rather than returning {}: a non-MLX response retires the verdict but
// leaves mlxKvBits absent, and a scenario that saw an empty object would keep
// a stale width alive and pass for the wrong reason.
function mlxRuntimeStateFrom(resp: any) {
  if (resp?.is_mlx !== true) {
    return {
      loadedMlxKvBitsRequested: null,
      mlxKvQuantReason: null,
      chatTemplateOverrideReason: null,
      mlxKvQuantNote: null,
    };
  }
  return {
    mlxKvBits: resp.mlx_kv_bits_requested ?? null,
    loadedMlxKvBitsRequested: resp.mlx_kv_bits_requested ?? null,
    mlxKvQuantReason: resp.mlx_kv_quant_reason ?? null,
    chatTemplateOverrideReason: resp.chat_template_override_reason ?? null,
    mlxKvQuantNote: resp.mlx_kv_quant_note ?? null,
  };
}
// Imported by chat-adapter.ts from ../utils/mmproj-fallback, and used by the auto-load
// success toast to say how a load was degraded. Without a stub it is a bare
// ReferenceError inside the retry loop, which scores as a failed load and fails every
// scenario as a wrong-model assertion -- the exact shape the guard below exists to
// catch, and the third time it has happened (see #7699).
//
// Mirrors the real composition rather than returning a fixed string: these scenarios do
// not assert on toast copy, but a stub that ignored its arguments would let a call site
// stop passing one of the two reasons without anything here noticing, which is the bug
// this helper was introduced to fix. The exact wording lives in mmproj-fallback.ts and
// is tested in studio/frontend/tests/mmproj-fallback.test.ts.
function loadFallbackNotice(
  baseTitle: string,
  cpuFallbackReason: any,
  mmprojFallbackReason: any,
) {
  const parts: string[] = [];
  if (cpuFallbackReason) parts.push(`cpu:${cpuFallbackReason}`);
  if (mmprojFallbackReason) parts.push(`mmproj:${mmprojFallbackReason}`);
  return {
    title: parts.length > 0 ? `${baseTitle} (${parts.join(", ")})` : baseTitle,
    description: parts.length > 0 ? parts.join(" ") : undefined,
    degraded: parts.length > 0,
  };
}
async function prepareHfTokenForUse(token: any) {
  EVENTS.push({ kind: "prepareHfToken" });
  if (SCENARIO.tokenDecision === "decline") return { proceed: false, token };
  if (SCENARIO.tokenDecision === "anonymous") return { proceed: true, token: null };
  return { proceed: true, token };
}
async function fetchGgufStagedMetadata(_req: any) {
  // camelCase, matching chat-api.ts: the call site reads `.isDiffusion`, so a
  // snake_case key here would be silently undefined.
  return {
    contextLength: null,
    layerCount: null,
    moeLayerCount: null,
    isDiffusion: false,
    diffusionUnknown: false,
  };
}

function syncModelCapabilities(modelId: string, resp: any) {
  const store = useChatRuntimeStore.getState();
  const models = store.models;
  const synced = {
    isVision: Boolean(resp.is_vision),
    isGguf: Boolean(resp.is_gguf),
    isAudio: Boolean(resp.is_audio),
    audioType: resp.audio_type ?? null,
    hasAudioInput: Boolean(resp.has_audio_input),
  };
  const idx = models.findIndex((m: any) => m.id === modelId);
  if (idx === -1) {
    store.setModels([
      ...models,
      { id: modelId, name: resp.display_name || modelId, isLora: Boolean(resp.is_lora), ...synced },
    ]);
  } else {
    const next = [...models];
    next[idx] = { ...next[idx], ...synced };
    store.setModels(next);
  }
}
const useChatRuntimeStore = {
  getState: () => STORE,
  // Recorded, not discarded: a scenario asserting on them has nowhere else to read them.
  setState: (p: any) => {
    Object.assign(STORE, typeof p === "function" ? p(STORE) : p);
  },
};

function createLoadingToastIcon() { return null; }
const toast: any = Object.assign(
  (_msg: string, _opts?: any) => "toast-id",
  {
    message: (msg: string, opts?: any) => {
      EVENTS.push({
        kind: "toast.message",
        msg,
        description: opts?.description,
        hasCancel: Boolean(opts?.action),
      });
      if (opts?.action?.onClick) CANCEL_TOAST_ACTION = opts.action.onClick;
      return "toast-id";
    },
    success: (msg: string) => EVENTS.push({ kind: "toast.success", msg }),
    error: (msg: string, opts?: any) =>
      EVENTS.push({ kind: "toast.error", msg, description: opts?.description }),
    dismiss: () => EVENTS.push({ kind: "toast.dismiss" }),
    info: (msg: string) => EVENTS.push({ kind: "toast.info", msg }),
  },
);

function mmprojFallbackMessage(reason: string) {
  return `mmproj fallback: ${reason}`;
}

async function tryAdoptServerActiveModel(options: any) {
  // Record adoption without reproducing store hydration.
  const id = options?.status?.active_model;
  if (!id) return false;
  EVENTS.push({ kind: "adoptServerModel", id });
  return true;
}
function resolveSpeculativeSettingsForLoad() {
  return { speculativeType: null, specDraftNMax: 0 };
}
function readLastLocalModelLoad() { return SCENARIO.lastLoaded; }
function recordLastLocalModelLoad(x: any) {
  EVENTS.push({ kind: "recordLastLocal", id: x.id, modelKind: x.kind });
}
function normalizeMaxSeqLength(value: any) {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}
function resolveInitialConfig(_id: string, _variant: any) {
  return { config: {
    customContextLength: null, maxSeqLength: null, gpuMemoryMode: null,
    gpuLayers: null, nCpuMoe: null, selectedGpuIds: undefined,
    speculativeType: null, specDraftNMax: null, chatTemplateOverride: null,
    kvCacheDtype: null, tensorParallel: false,
    ...(SCENARIO.config ?? {}),
  } };
}
// Mirrors the real predicate rather than returning a constant: the sliced region
// stores its result as the load's context pin, so a stub that always answered null
// would make every autoload scenario here agree with a bug in it. It takes the
// user's own Context Length (config.customContextLength above), never the n_ctx
// that went on the wire, which is Auto-resolved on a same-model reload.
function resolveExplicitCtxPin(n: any) { return n && n > 0 ? n : null; }
// Mirrors the real resolution: the pin, else the sentinel for a self-sizing backend and
// the app default for one that does not, which is what these off-Mac scenarios send.
function resolveLoadMaxSeqLength(args: any) {
  return (
    args.customContextLength ??
    args.pinnedMaxSeqLength ??
    (args.isGguf || args.isMlx ? 0 : args.defaultMaxSeqLength)
  );
}
// The window reported, never the request: the sentinel is below the control's minimum.
function loadedContextForParams(reported: any, requested: number, previous: number) {
  return reported ?? (requested > 0 ? requested : previous);
}
function resolveFitMaxSeqLength(
  isGguf: any, gpuMemoryMode: string, gpuLayers: number, pin: number | null, fallback: number,
) {
  if (!isGguf || gpuMemoryMode !== "manual" || gpuLayers >= 0) return fallback;
  return pin && pin > 0 ? pin : 0;
}
function localMaxTokensCeiling(loadedContextLength: number | null, fallback: number) {
  return Math.max(64, loadedContextLength ?? fallback);
}
function replayMaxTokensCap(loadedContextLength: number | null | undefined) {
  return loadedContextLength == null ? undefined : Math.max(64, loadedContextLength);
}
function unreportedWindowMaxTokens(g: boolean, held: number) { return g ? held : 4096; }
function resolveManualAutoCtxPin(..._a: any[]) { return null; }
async function ensureGpuDeviceCache() {}
function reconcilePersistedGpuIds(ids: any) { return ids; }
function saveSpeculativeType(_x: any) {}
function persistGpuMemoryModeOnLoad(..._a: any[]) {}
function reasoningCapsFromLoad(_x: any) { return {}; }
function resolveToolsEnabledOnLoad(_x: any) { return {}; }
function loadedGpuMemoryFields(_x: any) { return {}; }
function resolveLoadedSpeculativeSettings(_x: any) { return {}; }
function isMultimodalResponse(_x: any) { return false; }

async function listCachedGguf(signal?: any) {
  EVENTS.push({ kind: "listCachedGguf", hasSignal: Boolean(signal) });
  if (SCENARIO.ggufRepos === "throw") throw new Error("cached gguf listing failed");
  return SCENARIO.ggufRepos as any;
}
async function listCachedModels(_token?: any, signal?: any) {
  EVENTS.push({ kind: "listCachedModels", hasSignal: Boolean(signal) });
  if (SCENARIO.modelRepos === "throw") throw new Error("cached model listing failed");
  return SCENARIO.modelRepos as any;
}
// The unified on-device inventory (models dir, LM Studio, custom scan folders).
// Before #7374 was fixed the cascade never asked for it, so a downloaded local
// model was invisible and Send fetched an unrelated default instead.
async function listLocalModels() {
  if (SCENARIO.localModels === "throw") throw new Error("local inventory failed");
  return { models: (SCENARIO.localModels ?? []) as any[] };
}
function isHiddenModelId(..._a: any[]) { return false; }
// A chat-only install (Mac / AMD / CPU) runs GGUF, plus MLX on a Mac; the
// picker hides every other local format there, so a background pick must too.
const usePlatformStore = {
  getState: () => ({
    deviceType: SCENARIO.platform?.deviceType ?? SCENARIO.deviceType ?? "linux",
    chatOnlyReason: SCENARIO.platform?.chatOnlyReason ?? null,
    // Server-reported unless a scenario says the probe has not landed.
    fetched: SCENARIO.platformFetched !== false,
    isChatOnly: () => SCENARIO.chatOnly === true,
  }),
};
function isMlxId(value: string) {
  return typeof value === "string" && value.toLowerCase().includes("mlx");
}

// --- Hub download manager -------------------------------------------------
// The default-model fetch is a managed job now, so the harness models its
// surface: a start request, terminal listeners, and a cancel.
const DOWNLOAD_KIND = { MODEL: "model", DATASET: "dataset" } as const;
function jobKeyOf(kind: string, repoId: string, variant: string | null) {
  return variant ? `${kind}:${repoId}#${variant}` : `${kind}:${repoId}`;
}
function selectActiveJob(state: any, kind: string, repoId: string, variant?: any) {
  const job = state.jobs[jobKeyOf(kind, repoId, variant ?? null)];
  return job && job.state !== "cancelled" ? job : null;
}
let DOWNLOAD_STATE: any = { jobs: {} };
let DOWNLOAD_SUBS: any[] = [];
let CANCELLED_KEYS = new Set<string>();
let CANCEL_TOAST_ACTION: any = null;
function notifyDownloadStore() {
  for (const listener of [...DOWNLOAD_SUBS]) listener(DOWNLOAD_STATE);
}
const useDownloadManagerStore = {
  getState: () => DOWNLOAD_STATE,
  subscribe: (listener: any) => {
    DOWNLOAD_SUBS.push(listener);
    return () => {
      DOWNLOAD_SUBS = DOWNLOAD_SUBS.filter((entry) => entry !== listener);
    };
  },
};
let DOWNLOAD_LISTENERS: any[] = [];
function subscribeJobListeners(_kind: string, _repoId: string, handlers: any) {
  DOWNLOAD_LISTENERS.push(handlers);
  return () => {
    DOWNLOAD_LISTENERS = DOWNLOAD_LISTENERS.filter((entry) => entry !== handlers);
  };
}
const downloadManager = {
  requestStart: async (request: any) => {
    EVENTS.push({
      kind: "download.start",
      repoId: request.repoId,
      variant: request.variant,
    });
    const behaviour = SCENARIO.download ? SCENARIO.download(request) : "complete";
    if (behaviour === "busy" || behaviour === "conflict" || behaviour === "error") {
      return behaviour;
    }
    // The real requestStart runs transport preflights before any job exists.
    // A scenario can click the toast's Cancel inside that window.
    if (SCENARIO.cancelDuringStart) CANCEL_TOAST_ACTION?.();
    const key = jobKeyOf(request.kind, request.repoId, request.variant);
    DOWNLOAD_STATE = {
      jobs: {
        ...DOWNLOAD_STATE.jobs,
        [key]: {
          key,
          variant: request.variant,
          state: "running",
          downloadedBytes: 0,
          expectedBytes: request.expectedBytes,
        },
      },
    };
    notifyDownloadStore();
    // Terminal events land on a later turn, as the real poll loop's do.
    queueMicrotask(() => {
      if (CANCELLED_KEYS.has(key)) return;
      for (const handlers of [...DOWNLOAD_LISTENERS]) {
        if (behaviour === "complete") {
          handlers.onComplete?.(request.variant, request.expectedBytes);
        } else if (behaviour === "cancelled") {
          handlers.onCancelled?.(request.variant);
        } else {
          handlers.onError?.(request.variant);
        }
      }
    });
    return "started";
  },
  cancel: async (key: string) => {
    EVENTS.push({ kind: "download.cancel", key });
    const job = DOWNLOAD_STATE.jobs[key];
    // The real cancelJob can fail its request, probe the transfer as still
    // running, and restore the job. Nothing is cancelled in that case.
    if (SCENARIO.cancelFails) {
      if (job) {
        DOWNLOAD_STATE = {
          jobs: { ...DOWNLOAD_STATE.jobs, [key]: { ...job, state: "running" } },
        };
        notifyDownloadStore();
      }
      return;
    }
    CANCELLED_KEYS.add(key);
    if (job) {
      DOWNLOAD_STATE = {
        jobs: { ...DOWNLOAD_STATE.jobs, [key]: { ...job, state: "cancelled" } },
      };
      notifyDownloadStore();
    }
    for (const handlers of [...DOWNLOAD_LISTENERS]) {
      handlers.onCancelled?.(job?.variant ?? null);
    }
  },
};
async function listGgufVariants(repoId: string, _b?: any, options?: any) {
  EVENTS.push({
    kind: "listGgufVariants",
    repoId,
    hasSignal: Boolean(options?.signal),
  });
  const entry = SCENARIO.variants[repoId];
  if (entry === "throw") throw new Error("variant listing failed");
  return entry ?? { variants: [] };
}
async function validateModel(payload: any) {
  const result = SCENARIO.validate(payload);
  if (result instanceof Error) throw result;
  return result;
}
async function loadModel(payload: any) {
  const result = SCENARIO.load(payload);
  EVENTS.push({
    kind: "loadModel",
    model_path: payload.model_path,
    gguf_variant: payload.gguf_variant ?? null,
    // GGUF and self-sizing sources send 0; a Transformers one the safetensors length.
    max_seq_length: payload.max_seq_length ?? null,
    rejected: result instanceof Error,
  });
  if (result instanceof Error) throw result;
  return result;
}

// The speech-only verdict, mirroring
// studio/frontend/src/features/chat/lib/speech-only-status.ts. Real logic rather than a
// neutral stub: a stub that always answered false would keep every scenario green if the
// queued path's guard were removed.
export function isSpeechOnlyStatus(status: any): boolean {
  return (
    Boolean(status?.is_audio) &&
    status?.audio_type !== "whisper" &&
    status?.audio_type !== "audio_vlm"
  );
}
"""

SCENARIO_HELPERS = """
    const GEMMA = {
      repo_id: "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
      load_id: "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
      cache_path:
        "/home/john-doe/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-qat-GGUF",
      size_bytes: 15800000000,
    };
    const GEMMA_VARIANTS = {
      variants: [{
        quant: "UD-Q4_K_XL",
        filename: "UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        downloaded: true,
        size_bytes: 15800000000,
      }],
    };
    const OOM =
      "Failed to load model: llama-server was stopped by the operating system " +
      "(signal 9), most likely out of memory.";
    const VALIDATE_OK = () => ({
      requires_trust_remote_code: false,
      requires_security_review: false,
      requires_transformers_upgrade: false,
    });
    const LOADED = (payload) => ({
      model: payload.model_path,
      is_gguf: true,
      context_length: 32768,
    });
    // A backend-indexed local GGUF: exactly the row the old cascade could not
    // see, so Send downloaded a model the user already had a substitute for.
    const LOCAL_GGUF = {
      id: "/home/john-doe/models/qwen3-1.7b-instruct-q4_k_m.gguf",
      load_id: "/home/john-doe/models/qwen3-1.7b-instruct-q4_k_m.gguf",
      path: "/home/john-doe/models/qwen3-1.7b-instruct-q4_k_m.gguf",
      display_name: "qwen3-1.7b-instruct-q4_k_m",
      source: "models_dir",
      model_format: "gguf",
      size_bytes: 1_100_000_000,
      capabilities: { can_chat: true, requires_variant: false },
    };
    const EXTERNAL = "external::openai/gpt-5";
    // A safetensors repo, which MLX serves on a Mac and transformers everywhere else.
    const QWEN = {
      repo_id: "unsloth/Qwen3.5-4B",
      load_id: "unsloth/Qwen3.5-4B",
      cache_path: "/home/john-doe/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B",
      size_bytes: 8000000000,
    };
    const MAC = { deviceType: "mac", chatOnlyReason: null };
    // What a self-sizing backend answers: the window it resolved, not the request.
    const SERVED = (n) => (payload) => ({
      model: payload.model_path,
      is_gguf: false,
      context_length: n,
      native_context_length: 262144,
      max_context_length: 262144,
    });
    const scenario = (over) => ({
      ggufRepos: [],
      modelRepos: [],
      localModels: [],
      chatOnly: false,
      deviceType: "linux",
      cancelDuringStart: false,
      cancelFails: false,
      tokenDecision: null,
      platformFetched: true,
      variants: {},
      lastLoaded: null,
      serverLoading: [],
      serverResident: null,
      statusFailures: 0,
      serverLoadingClearsAfter: null,
      visibleCheckpoint: "",
      validate: VALIDATE_OK,
      load: LOADED,
      download: () => "complete",
      ...over,
    });
"""


def _require_node():
    if shutil.which("node") is None:
        pytest.skip("node not available")
    if not ADAPTER.exists():
        pytest.skip("studio chat sources not present")
    try:
        result = subprocess.run(
            ["node", "--experimental-strip-types", "--version"],
            capture_output = True,
            text = True,
            # A cold Windows runner is slow to start node; an impatient probe would fail the gate.
            timeout = 60,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("node could not be started")
    if result.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")


def _build_harness(run_dir: Path):
    """Slice autoLoadSmallestModel and its helpers verbatim out of the adapter."""
    lines = ADAPTER.read_text(encoding = "utf-8").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith("const MAX_AUTO_LOAD_ATTEMPTS")),
        None,
    )
    end = next(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith("export function createOpenAIStreamAdapter")
        ),
        None,
    )
    assert (
        start is not None and end is not None and start < end
    ), "could not locate the auto-load region in chat-adapter.ts"
    body = "\n".join(lines[start:end])
    assert "async function autoLoadSmallestModel" in body
    # Anything the adapter imports and the sliced region uses has to exist in the
    # preamble. Otherwise it is a bare ReferenceError at runtime, the retry loop
    # catches it and scores it as a failed load, and the scenario fails as a
    # wrong-model assertion that says nothing about the real cause. That is what
    # #7699 did by adding a syncModelCapabilities call here.
    imported = set()
    source = "\n".join(lines)
    for match in re.finditer(
        r"^import\s+(?:\*\s+as\s+)?([A-Za-z_$][\w$]*)\s*(?:,|from)", source, re.M
    ):
        imported.add(match.group(1))
    # Default and namespace forms bind a name too, and it is the same ReferenceError when the harness lacks it.
    # chat-adapter.ts has none today, so this is for the first one somebody adds.
    for match in re.finditer(
        r"^import\s+(?:[A-Za-z_$][\w$]*\s*,\s*)?\{([^}]*)\}\s+from", source, re.M
    ):
        for spec in match.group(1).split(","):
            spec = spec.strip()
            # `import { type Foo }` is erased at runtime, so it can never be a
            # ReferenceError and must not be demanded of the harness.
            if not spec or spec.startswith("type "):
                continue
            name = spec.split(" as ")[-1].strip()
            if name:
                imported.add(name)
    # Any mention, not just a call. useChatRuntimeStore, toast and GPU_LAYERS_AUTO are all used in this region without a
    # following paren, and each would be the same ReferenceError; they pass today only because the preamble happens to
    # define them. Comments are stripped first so a name discussed in prose does not count as a use.
    code = re.sub(r"/\*.*?\*/", "", body, flags = re.S)
    code = re.sub(r"//[^\n]*", "", code)
    preamble = PREAMBLE + PONYFILLS + _wait_gate_source()
    missing = sorted(
        name
        for name in imported
        if re.search(rf"\b{re.escape(name)}\b", code)
        and not re.search(
            rf"^(?:export\s+)?(?:async\s+)?"
            rf"(?:function\s+|const\s+|let\s+|var\s+|class\s+){re.escape(name)}\b",
            preamble,
            re.M,
        )
    )
    assert not missing, (
        f"the sliced region uses {missing}, imported by chat-adapter.ts but absent "
        "from this harness. Add a stub to PREAMBLE, or these scenarios will fail as "
        "wrong-model assertions rather than saying what is actually missing."
    )
    (run_dir / "harness.ts").write_text(
        "// @ts-nocheck\n"
        + preamble
        + "\n"
        + body
        + "\nexport { autoLoadSmallestModel, resolveQueuedEmptyLocalModel };\n"
        + "export function storeState() { return STORE; }\n",
        encoding = "utf-8",
    )


def _run(
    scenario_expr: str,
    prelude: str = "",
    *,
    queued: bool = False,
) -> dict:
    _require_node()
    # Its own directory per invocation: a shared file lets one runner read another's rewrite.
    TEMP.mkdir(parents = True, exist_ok = True)
    run_dir = Path(tempfile.mkdtemp(prefix = "run", dir = TEMP))
    _build_harness(run_dir)
    # The real send path always supplies an abort signal, so every scenario
    # exercises the signal plumbing whichever entry point it enters through.
    entry = (
        "resolveQueuedEmptyLocalModel(signal)"
        if queued
        else "autoLoadSmallestModel({ abortSignal: signal })"
    )
    script = (
        textwrap.dedent(
            """
        // @ts-nocheck
        import {
          autoLoadSmallestModel,
          resolveQueuedEmptyLocalModel,
          setScenario,
          EVENTS,
          storeState,
        } from "./harness.ts";
        """
        )
        + SCENARIO_HELPERS
        + textwrap.dedent(prelude)
        + textwrap.dedent(
            f"""
        setScenario({scenario_expr});
        const signal = new AbortController().signal;
        const result = await {entry};
        const s = storeState();
        console.log(JSON.stringify({{
          result,
          events: EVENTS,
          store: {{
            maxSeqLength: s.params?.maxSeqLength ?? null,
            maxTokens: s.params?.maxTokens ?? null,
            customContextLength: s.customContextLength ?? null,
            loadedCustomContextLength: s.loadedCustomContextLength ?? null,
            loadedContextLength: s.loadedContextLength ?? null,
          }},
        }}));
        """
        )
    )
    (run_dir / "run.mts").write_text(script, encoding = "utf-8")
    completed = subprocess.run(
        ["node", "--experimental-strip-types", "--no-warnings", "run.mts"],
        cwd = str(run_dir),
        capture_output = True,
        text = True,
        # Explicit: text alone decodes with the Windows ANSI code page, which mangles the non-ASCII toast copy node
        # emits as UTF-8.
        encoding = "utf-8",
        timeout = 60,
        env = dict(os.environ, NODE_NO_WARNINGS = "1"),
    )
    assert completed.returncode == 0, f"stderr: {completed.stderr}\nstdout: {completed.stdout}"
    last = [line for line in completed.stdout.strip().splitlines() if line.strip()][-1]
    return json.loads(last)


def _loaded_paths(out: dict) -> list[str]:
    return [event["model_path"] for event in out["events"] if event["kind"] == "loadModel"]


def _toasts(out: dict, kind: str) -> list[dict]:
    return [event for event in out["events"] if event["kind"] == kind]


def _downloads_started(out: dict) -> list[str]:
    return [event["repoId"] for event in out["events"] if event["kind"] == "download.start"]


def test_failed_cached_load_does_not_download_the_default_model():
    """The reported case: the only cached repo enumerates fine but its load OOMs, so auto-load stops
    there rather than fetch a model the user never asked for."""
    out = _run(
        "scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " load: (p) => p.model_path === GEMMA.repo_id ? new Error(OOM) : LOADED(p) })"
    )

    assert _loaded_paths(out) == ["unsloth/gemma-4-26B-A4B-it-qat-GGUF"]
    assert DEFAULT_MODEL not in _loaded_paths(out)
    assert out["result"]["loaded"] is False
    assert _toasts(out, "toast.success") == []
    assert not any(
        "Downloading a small model" in event["msg"] for event in _toasts(out, "toast.message")
    )
    assert _downloads_started(out) == []


def test_failed_cached_load_surfaces_the_backend_reason():
    out = _run(
        "scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " load: () => new Error(OOM) })"
    )

    [error] = _toasts(out, "toast.error")
    assert error["msg"] == "Could not load unsloth/gemma-4-26B-A4B-it-qat-GGUF (UD-Q4_K_XL)"
    assert "out of memory" in error["description"]


def test_load_rejection_without_a_message_still_names_the_model():
    """Old backends and non-Error throws carry no detail; still name the model, not the default."""
    out = _run(
        "scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " load: () => new Error('') })"
    )

    assert DEFAULT_MODEL not in _loaded_paths(out)
    [error] = _toasts(out, "toast.error")
    assert "unsloth/gemma-4-26B-A4B-it-qat-GGUF (UD-Q4_K_XL)" in error["msg"]
    assert error["description"]


def test_empty_device_still_downloads_the_default_model():
    """Nothing cached means nothing failed, so the download path is untouched."""
    out = _run("scenario({})")

    assert _loaded_paths(out) == [DEFAULT_MODEL]
    assert out["result"]["loaded"] is True
    assert _toasts(out, "toast.error") == []
    assert [event["msg"] for event in _toasts(out, "toast.success")] == [
        "Loaded Gemma 4 E2B (UD-Q4_K_XL)"
    ]


def test_enumeration_failure_still_downloads_the_default_model():
    """A repo whose variants cannot be listed never reached /load, so it keeps falling through."""
    out = _run("scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: 'throw' } })")

    assert _loaded_paths(out) == [DEFAULT_MODEL]
    assert out["result"]["loaded"] is True


def test_consent_gated_candidate_still_downloads_the_default_model():
    """trust_remote_code / security review block before the load: a deferral, not a failure."""
    out = _run(
        "scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " validate: (p) => p.model_path === GEMMA.repo_id"
        " ? { requires_trust_remote_code: true, requires_security_review: false,"
        " requires_transformers_upgrade: false } : VALIDATE_OK() })"
    )

    assert _loaded_paths(out) == [DEFAULT_MODEL]
    assert out["result"]["loaded"] is True


def test_attempt_cap_still_gates_the_default_download():
    """Four broken repos: the sweep tries smaller candidates, stops at three, never reaches the Hub."""
    out = _run(
        "scenario({ ggufRepos: [1, 2, 3, 4].map((i) => ({ ...GEMMA, repo_id: `r${i}`,"
        " load_id: `r${i}`, size_bytes: i })),"
        " variants: Object.fromEntries([1, 2, 3, 4].map((i) => [`r${i}`, GEMMA_VARIANTS])),"
        " load: () => new Error(OOM) })"
    )

    assert _loaded_paths(out) == ["r1", "r2", "r3"]
    assert DEFAULT_MODEL not in _loaded_paths(out)


def test_a_later_cached_model_can_still_load_after_an_earlier_failure():
    """One broken repo must not veto a working one; the failure toast is only for an empty sweep."""
    out = _run(
        "scenario({ ggufRepos: [1, 2].map((i) => ({ ...GEMMA, repo_id: `r${i}`,"
        " load_id: `r${i}`, size_bytes: i })),"
        " variants: { r1: GEMMA_VARIANTS, r2: GEMMA_VARIANTS },"
        " load: (p) => p.model_path === 'r1' ? new Error(OOM) : LOADED(p) })"
    )

    assert _loaded_paths(out) == ["r1", "r2"]
    assert out["result"]["loaded"] is True
    assert _toasts(out, "toast.error") == []


def test_reported_failure_is_flagged_so_callers_drop_the_generic_advice():
    """Both send paths toast a generic "No model loaded" on loaded: false, burying the detailed one."""
    out = _run(
        "scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " load: () => new Error(OOM) })"
    )

    assert out["result"]["loaded"] is False
    assert out["result"]["loadFailureReported"] is True


def test_empty_device_does_not_flag_a_reported_failure():
    """Nothing was attempted, so the callers must keep their generic advice."""
    out = _run("scenario({ ggufRepos: [], models: [] })")

    assert out["result"].get("loadFailureReported") is not True


def test_a_resume_only_cached_row_is_skipped_so_the_default_still_downloads():
    """An interrupted download leaves a partial / can_chat=false row for resume and delete; sweeping
    it anyway would let its rejection suppress the default download."""
    out = _run(
        "scenario({ modelRepos: [{ repo_id: 'org/half', load_id: 'org/half',"
        " size_bytes: 1, partial: true, capabilities: { can_chat: false } }],"
        " load: (p) => p.model_path === 'org/half'"
        " ? new Error('config.json not found') : LOADED(p) })"
    )

    assert _loaded_paths(out) == [DEFAULT_MODEL]
    assert out["result"]["loaded"] is True
    assert _toasts(out, "toast.error") == []


def test_a_can_chat_false_cached_row_is_skipped_on_its_own():
    """The two fields are set independently, so can_chat alone must be enough to skip a row."""
    out = _run(
        "scenario({ ggufRepos: [{ ...GEMMA, capabilities: { can_chat: false } }],"
        " variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " load: (p) => p.model_path === GEMMA.repo_id ? new Error(OOM) : LOADED(p) })"
    )

    assert _loaded_paths(out) == [DEFAULT_MODEL]
    assert out["result"]["loaded"] is True


def test_the_last_used_model_is_skipped_when_its_row_went_partial():
    """The last-used shortcut reads the same rows, so a cancelled update must not spend an attempt."""
    out = _run(
        "scenario({ ggufRepos: [{ ...GEMMA, partial: true }],"
        " variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " lastLoaded: { id: GEMMA.repo_id, kind: 'gguf', ggufVariant: 'UD-Q4_K_XL' },"
        " load: (p) => p.model_path === GEMMA.repo_id ? new Error(OOM) : LOADED(p) })"
    )

    assert _loaded_paths(out) == [DEFAULT_MODEL]
    assert out["result"]["loaded"] is True


def test_a_complete_cached_row_is_still_attempted():
    """Guard on the filter: a healthy row is still swept, and an older backend omitting the fields
    keeps its behaviour."""
    out = _run(
        "scenario({ ggufRepos: [{ ...GEMMA, partial: false, capabilities: { can_chat: true } }],"
        " variants: { [GEMMA.repo_id]: GEMMA_VARIANTS } })"
    )

    assert _loaded_paths(out) == [GEMMA_REPO]
    assert out["result"]["loaded"] is True


def test_a_rejected_validation_does_not_download_the_default_model():
    """A cached candidate can be refused by /validate rather than /load. The sweep's catches are
    bare, so an unrecorded rejection reads as an empty device and fetches an unasked-for model."""
    out = _run(
        "scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " validate: (p) => p.model_path === GEMMA.repo_id"
        " ? new Error('snapshot no longer exists') : ({}) })"
    )

    assert DEFAULT_MODEL not in _loaded_paths(out)
    assert out["result"]["loaded"] is False
    assert out["result"]["loadFailureReported"] is True
    [error] = _toasts(out, "toast.error")
    assert error["msg"] == "Could not load unsloth/gemma-4-26B-A4B-it-qat-GGUF (UD-Q4_K_XL)"
    assert "snapshot no longer exists" in error["description"]


def test_a_rejected_validation_still_lets_a_later_cached_model_load():
    """Control for the test above: recording must not end the sweep; only a declined dialog does."""
    out = _run(
        "scenario({ ggufRepos: [1, 2].map((i) => ({ ...GEMMA, repo_id: `r${i}`,"
        " load_id: `r${i}`, size_bytes: i })),"
        " variants: { r1: GEMMA_VARIANTS, r2: GEMMA_VARIANTS },"
        " validate: (p) => p.model_path === 'r1'"
        " ? new Error('snapshot no longer exists') : ({}) })"
    )

    assert _loaded_paths(out) == ["r2"]
    assert out["result"]["loaded"] is True
    assert _toasts(out, "toast.error") == []


# #7374: a model already on disk must be found before anything is fetched


# --- #7374: a model already on disk must be found before anything is fetched ---
def test_an_indexed_local_gguf_loads_instead_of_downloading_the_default():
    """The reported bug. The user has a GGUF in their models dir, but the cascade
    only read the two managed-cache lists, so Send fetched a model instead."""
    out = _run("scenario({ localModels: [LOCAL_GGUF] })")

    assert _loaded_paths(out) == [LOCAL_GGUF_PATH]
    assert _downloads_started(out) == []
    assert out["result"]["loaded"] is True


def test_a_local_model_wins_over_the_default_even_when_the_cache_is_empty():
    """An empty managed cache is not an empty device."""
    out = _run("scenario({ ggufRepos: [], modelRepos: [], localModels: [LOCAL_GGUF] })")

    assert DEFAULT_MODEL not in _loaded_paths(out)
    assert _downloads_started(out) == []


def test_the_smallest_on_device_model_wins_across_inventories():
    """Cached and local rows compete in one size-ordered cascade, not one list
    then the other."""
    out = _run(
        "scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " localModels: [LOCAL_GGUF] })"
    )

    # LOCAL_GGUF is 1.1 GB, GEMMA is 15.8 GB.
    assert _loaded_paths(out) == [LOCAL_GGUF_PATH]


@pytest.mark.parametrize(
    "inventory",
    ["ggufRepos", "modelRepos", "localModels"],
)
def test_an_inventory_failure_never_reads_as_an_empty_device(inventory):
    """Both cached lookups used to be wrapped in `.catch(() => [])`, so one flaky
    request looked exactly like "this user has no models" and started a download."""
    out = _run(f"scenario({{ {inventory}: 'throw' }})")

    assert _downloads_started(out) == []
    assert _loaded_paths(out) == []
    assert out["result"]["loaded"] is False


@pytest.mark.parametrize("broken", ["ggufRepos", "modelRepos", "localModels"])
def test_one_broken_inventory_still_loads_a_model_another_one_found(broken):
    """A failure is one unknown source, not a verdict on the other two.
    Promise.all rejected the batch, discarding lists that had already arrived,
    so a broken local scan left a loadable model unused."""
    # Put the model in a source that is not the one being broken.
    holder = "localModels" if broken != "localModels" else "ggufRepos"
    rows = "[LOCAL_GGUF]" if holder == "localModels" else "[GEMMA]"
    extra = "" if holder == "localModels" else ", variants: { [GEMMA.repo_id]: GEMMA_VARIANTS }"
    out = _run(f"scenario({{ {broken}: 'throw', {holder}: {rows}{extra} }})")

    assert out["result"]["loaded"] is True
    # Still fails closed on the part it cannot see: no default is fetched.
    assert _downloads_started(out) == []
    assert DEFAULT_MODEL not in _loaded_paths(out)


def test_a_local_row_the_picker_would_hide_is_never_auto_loaded():
    """A background pick must be something the user could have picked themselves."""
    hidden = (
        "{ ...LOCAL_GGUF, capabilities: { can_chat: false } },"
        " { ...LOCAL_GGUF, id: 'p', load_id: 'p', path: 'p', partial: true },"
        " { ...LOCAL_GGUF, id: 'a', load_id: 'a', path: 'a', model_format: 'adapter' },"
        " { ...LOCAL_GGUF, id: 'c', load_id: 'c', path: 'c', model_format: 'checkpoint' },"
        " { ...LOCAL_GGUF, id: 'o', load_id: 'o', path: 'o', source: 'ollama' }"
    )
    out = _run(f"scenario({{ localModels: [{hidden}] }})")

    assert _loaded_paths(out) == [DEFAULT_MODEL]
    assert _downloads_started(out) == [DEFAULT_MODEL]


def test_the_remembered_local_model_is_tried_before_a_smaller_one():
    """Re-picking the model the user actually ran beats re-picking the smallest."""
    out = _run(
        "scenario({ localModels: [LOCAL_GGUF,"
        " { ...LOCAL_GGUF, id: 'tiny', load_id: 'tiny', path: 'tiny', size_bytes: 1 }],"
        " lastLoaded: { id: LOCAL_GGUF.load_id, kind: 'gguf', ggufVariant: null } })"
    )

    assert _loaded_paths(out) == [LOCAL_GGUF_PATH]


def test_the_default_model_is_fetched_through_the_download_manager():
    """Not inline inside /api/inference/load: the manager owns the transfer, so
    it gets a panel entry, progress, and a Cancel that stops it."""
    out = _run("scenario({})")

    assert _downloads_started(out) == [DEFAULT_MODEL]
    # The bytes land before the load is attempted.
    kinds = [event["kind"] for event in out["events"]]
    assert kinds.index("download.start") < kinds.index("loadModel")
    # Nothing loadable on device, so the default download is the correct answer.
    assert _loaded_paths(out) == [DEFAULT_MODEL]


def test_cancelling_the_first_download_loads_nothing_and_says_what_to_do():
    out = _run("scenario({ download: () => 'cancelled' })")

    assert _loaded_paths(out) == []
    assert out["result"]["loaded"] is False
    [notice] = [
        event for event in _toasts(out, "toast.message") if "download stopped" in event["msg"]
    ]
    assert "Pick one from the top bar" in notice["description"]


def test_a_failed_first_download_never_reaches_the_load():
    """A dead transfer must not be handed to /load as if the file were there."""
    for behaviour in ("failed", "error", "busy", "conflict"):
        out = _run(f"scenario({{ download: () => '{behaviour}' }})")
        assert _loaded_paths(out) == [], behaviour
        assert out["result"]["loaded"] is False, behaviour


def test_the_first_download_toast_explains_rather_than_alarms():
    """The old copy read as a warning about the user's machine and named a raw
    repo id and quant."""
    out = _run("scenario({})")

    messages = [event["msg"] for event in _toasts(out, "toast.message")]
    descriptions = [event["description"] or "" for event in _toasts(out, "toast.message")]
    assert any("Getting Gemma 4 E2B ready" in msg for msg in messages)
    assert any(
        "Unsloth couldn\u2019t find an existing model. Unsloth is now getting "
        "Gemma 4 E2B ready for use. You can stop the download or manage models "
        "later in the 'Model hub'" == text
        for text in descriptions
    )
    assert not any("No downloaded models found" in text for text in descriptions)


def test_a_chat_only_install_never_auto_loads_a_format_it_cannot_run():
    """On chat-only builds the picker shows GGUF (and MLX on a Mac) only, so a
    background pick must not spend an attempt on a safetensors row."""
    safetensors = (
        "{ ...LOCAL_GGUF, id: 'st', load_id: 'st', path: '/models/st',"
        " model_format: 'safetensors' }"
    )
    out = _run(f"scenario({{ chatOnly: true, localModels: [{safetensors}] }})")
    assert "st" not in _loaded_paths(out)
    assert _loaded_paths(out) == [DEFAULT_MODEL]

    # The same row is fair game on a full install.
    out = _run(f"scenario({{ chatOnly: false, localModels: [{safetensors}] }})")
    assert _loaded_paths(out) == ["st"]


def test_a_mac_chat_only_install_still_auto_loads_a_local_mlx_model():
    mlx = (
        "{ ...LOCAL_GGUF, id: 'org/Qwen3-4B-mlx-4bit', load_id: '/models/mlx',"
        " path: '/models/mlx', model_format: 'safetensors' }"
    )
    out = _run(f"scenario({{ chatOnly: true, deviceType: 'mac', localModels: [{mlx}] }})")
    assert _loaded_paths(out) == ["/models/mlx"]


def test_an_image_generation_row_is_never_auto_loaded_for_chat():
    """Diffusion rows carry an image/video task while can_chat stays true on
    file format alone. The picker routes them to the Images page on click; a
    background load has no routing step, so it has to skip them."""
    for task in ("text-to-image", "text-to-video", "image-diffusion-unsupported"):
        rows = (
            f"{{ ...LOCAL_GGUF, id: 'img', load_id: 'img', path: 'img', task: '{task}',"
            " size_bytes: 1 }, LOCAL_GGUF"
        )
        out = _run(f"scenario({{ localModels: [{rows}] }})")
        # 'img' is the smaller row, so only the task gate can keep it out.
        assert _loaded_paths(out) == [LOCAL_GGUF_PATH], task


def test_a_chat_row_with_no_task_tag_still_auto_loads():
    """Control for the gate above: the backend leaves task null for chat LLMs."""
    out = _run("scenario({ localModels: [{ ...LOCAL_GGUF, task: null }] })")
    assert _loaded_paths(out) == [LOCAL_GGUF_PATH]


def test_a_failed_quant_falls_through_to_the_next_one_in_the_same_repo():
    """One corrupt quant must not cost a repo that holds a valid one."""
    variants = (
        "{ variants: ["
        "{ quant: 'Q2_K', filename: 'm-Q2_K.gguf', downloaded: true, size_bytes: 100 },"
        "{ quant: 'Q4_K_M', filename: 'm-Q4_K_M.gguf', downloaded: true, size_bytes: 200 }"
        "] }"
    )
    out = _run(
        f"scenario({{ ggufRepos: [GEMMA], variants: {{ [GEMMA.repo_id]: {variants} }},"
        " load: (p) => p.gguf_variant === 'Q2_K' ? new Error(OOM) : LOADED(p) })"
    )

    assert [event["gguf_variant"] for event in out["events"] if event["kind"] == "loadModel"] == [
        "Q2_K",
        "Q4_K_M",
    ]
    assert out["result"]["loaded"] is True
    assert _downloads_started(out) == []


def test_the_variant_retry_still_respects_the_attempt_cap():
    """Guard on the loop above: a repo of broken quants must not spin."""
    variants = (
        "{ variants: [1,2,3,4,5,6].map((i) => ({ quant: `Q${i}`,"
        " filename: `m-Q${i}.gguf`, downloaded: true, size_bytes: i })) }"
    )
    out = _run(
        f"scenario({{ ggufRepos: [GEMMA], variants: {{ [GEMMA.repo_id]: {variants} }},"
        " load: () => new Error(OOM) })"
    )

    assert len(_loaded_paths(out)) == 3
    assert _downloads_started(out) == []


def test_cached_inventory_lookups_carry_the_run_abort_signal():
    """Both are raw fetches with no timeout of their own, so a stall would hold
    the model-loading lease open after the user stops the send."""
    out = _run("scenario({})")
    for kind in ("listCachedGguf", "listCachedModels"):
        [call] = [event for event in out["events"] if event["kind"] == kind]
        assert call["hasSignal"] is True, kind


def test_cancel_during_the_download_preflight_still_stops_it():
    """requestStart runs transport preflights before the job exists and cancel
    no-ops on a missing key, so the first Cancel click was swallowed."""
    out = _run("scenario({ cancelDuringStart: true })")

    assert [event["kind"] for event in out["events"]].count("download.cancel") >= 1
    assert _loaded_paths(out) == []
    assert out["result"]["loaded"] is False


def test_the_download_toast_carries_a_cancel_action():
    out = _run("scenario({})")
    assert any(
        event.get("hasCancel") and "Getting Gemma 4 E2B ready" in event["msg"]
        for event in _toasts(out, "toast.message")
    )


def test_remembered_posix_paths_match_case_sensitively():
    """Two local models on Linux can differ only by path casing; folding them
    marks both as the remembered one and can auto-load the wrong model."""
    rows = (
        "{ ...LOCAL_GGUF, id: '/models/Foo.gguf', load_id: '/models/Foo.gguf',"
        " path: '/models/Foo.gguf', size_bytes: 200 },"
        " { ...LOCAL_GGUF, id: '/models/foo.gguf', load_id: '/models/foo.gguf',"
        " path: '/models/foo.gguf', size_bytes: 100 }"
    )
    out = _run(
        f"scenario({{ localModels: [{rows}],"
        " lastLoaded: { id: '/models/Foo.gguf', kind: 'gguf', ggufVariant: null } })"
    )

    # Lowercasing would rank both as remembered and let the smaller one win.
    assert _loaded_paths(out) == ["/models/Foo.gguf"]


def test_a_mixed_cached_repo_is_attempted_once_not_twice():
    """A repo holding both GGUF and safetensors appears in both cached lists,
    but the backend resolves one load target to one model, so keeping both rows
    spends a second attempt on the same files."""
    out = _run(
        "scenario({ ggufRepos: [GEMMA],"
        " modelRepos: [{ repo_id: GEMMA.repo_id, load_id: GEMMA.load_id, size_bytes: 15800000000 }],"
        " variants: { [GEMMA.repo_id]: GEMMA_VARIANTS },"
        " load: () => new Error(OOM) })"
    )

    assert _loaded_paths(out) == [GEMMA_REPO]
    # The GGUF row is the one the backend will actually resolve, so it survives.
    assert [event["gguf_variant"] for event in out["events"] if event["kind"] == "loadModel"] == [
        "UD-Q4_K_XL"
    ]


def test_a_mixed_cached_repo_still_loads_through_its_gguf_row():
    """Control: deduping must not drop the repo altogether."""
    out = _run(
        "scenario({ ggufRepos: [GEMMA],"
        " modelRepos: [{ repo_id: GEMMA.repo_id, load_id: GEMMA.load_id, size_bytes: 1 }],"
        " variants: { [GEMMA.repo_id]: GEMMA_VARIANTS } })"
    )
    assert _loaded_paths(out) == [GEMMA_REPO]
    assert out["result"]["loaded"] is True


def test_distinct_repos_are_not_collapsed_by_the_dedupe():
    """Guard on the key: only a shared load target may collapse."""
    out = _run(
        "scenario({ ggufRepos: [1, 2].map((i) => ({ ...GEMMA, repo_id: `r${i}`,"
        " load_id: `r${i}`, size_bytes: i })),"
        " variants: { r1: GEMMA_VARIANTS, r2: GEMMA_VARIANTS },"
        " load: () => new Error(OOM) })"
    )
    assert _loaded_paths(out) == ["r1", "r2"]


def test_variant_scans_carry_the_run_abort_signal():
    """Bounded by their own timeout, but a stopped send should not wait it out."""
    out = _run("scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS } })")
    scans = [event for event in out["events"] if event["kind"] == "listGgufVariants"]
    assert scans, "no variant scan ran"
    assert all(scan["hasSignal"] for scan in scans)


def test_a_failed_path_does_not_mark_a_case_distinct_sibling_as_tried():
    """On Linux /models/Foo.gguf and /models/foo.gguf are two models, so folding
    the tried-candidate key let one failure suppress the other."""
    rows = (
        "{ ...LOCAL_GGUF, id: '/models/Foo.gguf', load_id: '/models/Foo.gguf',"
        " path: '/models/Foo.gguf', size_bytes: 1 },"
        " { ...LOCAL_GGUF, id: '/models/foo.gguf', load_id: '/models/foo.gguf',"
        " path: '/models/foo.gguf', size_bytes: 2 }"
    )
    out = _run(
        f"scenario({{ localModels: [{rows}],"
        " load: (p) => p.model_path === '/models/Foo.gguf' ? new Error(OOM) : LOADED(p) })"
    )

    assert _loaded_paths(out) == ["/models/Foo.gguf", "/models/foo.gguf"]
    assert out["result"]["loaded"] is True
    assert _downloads_started(out) == []


def test_the_default_variant_lookup_carries_the_run_abort_signal():
    """Bounded by its own 30s timeout, but a stopped send should not hold the
    model-loading lease waiting it out before the download even starts."""
    out = _run("scenario({})")
    [lookup] = [
        event
        for event in out["events"]
        if event["kind"] == "listGgufVariants" and event["repoId"] == DEFAULT_MODEL
    ]
    assert lookup["hasSignal"] is True


def test_a_download_that_survives_a_failed_cancel_is_not_loaded():
    """cancelJob can fail and the transfer keep running, but the user asked for
    it to stop, so the bytes that arrive anyway must not reach a load."""
    out = _run("scenario({ cancelDuringStart: true, cancelFails: true })")

    assert _loaded_paths(out) == []
    assert out["result"]["loaded"] is False


def test_a_failed_cancel_does_not_latch_the_toast_action_off():
    """The retry latch must clear once the request settles, or every later
    Cancel click is swallowed."""
    src = (WORKDIR / "studio/frontend/src/features/chat/api/chat-adapter.ts").read_text(
        encoding = "utf-8"
    )
    helper = src.split("async function ensureDefaultModelDownloaded", 1)[1]
    helper = helper.split("async function autoLoadSmallestModel", 1)[0]
    # In flight only, cleared when the request settles.
    assert 'if (cancelInFlight || active.state === "cancelling") return true;' in helper
    assert "cancelInFlight = false;\n    });" in helper
    # The subscription fires the deferred attempt once; retries come from clicks.
    assert "if (!cancelEverIssued) issueCancel();" in helper


def test_an_expired_token_is_prepared_before_the_managed_start():
    """startJob sends the stored token raw, with none of the recovery
    validateModel and loadModel get, so an expired one would fail the download
    of a public repo the lookup just read anonymously."""
    out = _run("scenario({})")
    kinds = [event["kind"] for event in out["events"]]
    assert "prepareHfToken" in kinds
    assert kinds.index("prepareHfToken") < kinds.index("download.start")


def test_declining_the_token_dialog_starts_no_download():
    out = _run("scenario({ tokenDecision: 'decline' })")
    assert _downloads_started(out) == []
    assert _loaded_paths(out) == []
    assert out["result"]["loaded"] is False


def test_the_token_is_not_prepared_when_the_default_is_already_on_disk():
    """Nothing to download means nothing to prompt about."""
    downloaded = (
        "{ variants: [{ quant: 'UD-Q4_K_XL', filename: 'm-UD-Q4_K_XL.gguf',"
        " downloaded: true, size_bytes: 100 }] }"
    )
    out = _run(f"scenario({{ variants: {{ 'unsloth/gemma-4-E2B-it-GGUF': {downloaded} }} }})")
    kinds = [event["kind"] for event in out["events"]]
    assert "prepareHfToken" not in kinds
    assert _downloads_started(out) == []
    assert _loaded_paths(out) == [DEFAULT_MODEL]


def test_a_cached_image_repo_is_never_auto_loaded_for_chat():
    """Cached diffusion repos report can_chat true on file format alone, and
    carry their Images/Video task on the row."""
    for task in ("text-to-image", "text-to-video", "image-diffusion-unsupported"):
        out = _run(
            f"scenario({{ ggufRepos: [{{ ...GEMMA, repo_id: 'img', load_id: 'img',"
            f" size_bytes: 1, task: '{task}' }}, GEMMA],"
            " variants: { img: GEMMA_VARIANTS, [GEMMA.repo_id]: GEMMA_VARIANTS } })"
        )
        assert _loaded_paths(out) == [GEMMA_REPO], task


def test_a_cached_text_generation_repo_still_auto_loads():
    """Chat GGUFs are tagged text-generation, not null, so the gate must be a
    list of image/video tasks rather than "has a task"."""
    out = _run(
        "scenario({ ggufRepos: [{ ...GEMMA, task: 'text-generation' }],"
        " variants: { [GEMMA.repo_id]: GEMMA_VARIANTS } })"
    )
    assert _loaded_paths(out) == [GEMMA_REPO]


def test_a_provisional_mac_platform_does_not_hide_a_remote_backends_models():
    """Before the probe lands chatOnly is a browser guess: a Mac browser on a
    remote Linux Unsloth would hide every local safetensors model."""
    safetensors = (
        "{ ...LOCAL_GGUF, id: 'st', load_id: 'st', path: '/models/st',"
        " model_format: 'safetensors' }"
    )
    out = _run(
        f"scenario({{ chatOnly: true, deviceType: 'mac', platformFetched: false,"
        f" localModels: [{safetensors}] }})"
    )

    assert _loaded_paths(out) == ["st"]
    assert _downloads_started(out) == []


def test_a_server_reported_chat_only_platform_still_gates():
    """Control: once the backend has answered, the gate applies."""
    safetensors = (
        "{ ...LOCAL_GGUF, id: 'st', load_id: 'st', path: '/models/st',"
        " model_format: 'safetensors' }"
    )
    out = _run(
        f"scenario({{ chatOnly: true, platformFetched: true, localModels: [{safetensors}] }})"
    )
    assert _loaded_paths(out) == [DEFAULT_MODEL]


def test_the_default_is_validated_before_its_download_starts():
    """Active training or the GPU placement guard can refuse the default, and
    learning that after several gigabytes costs a long wait for nothing."""
    out = _run(
        "scenario({ validate: () => ({ requires_trust_remote_code: false,"
        " requires_security_review: false, requires_transformers_upgrade: true }) })"
    )

    assert _downloads_started(out) == []
    assert _loaded_paths(out) == []
    assert out["result"]["loaded"] is False


def test_a_cleared_default_still_downloads_and_loads():
    """Control: the preflight must not block the normal path."""
    out = _run("scenario({})")
    kinds = [event["kind"] for event in out["events"]]
    assert kinds.index("download.start") < kinds.index("loadModel")
    assert _loaded_paths(out) == [DEFAULT_MODEL]


def test_a_chat_only_backend_skips_cached_non_gguf_rows():
    """The picker hides cached non-GGUF rows outright in chat-only mode, so the
    cascade must not auto-select a format the UI calls unrunnable."""
    out = _run(
        "scenario({ chatOnly: true, platformFetched: true,"
        " modelRepos: [{ repo_id: 'org/st', load_id: 'org/st', size_bytes: 1 }] })"
    )
    assert "org/st" not in _loaded_paths(out)
    assert _loaded_paths(out) == [DEFAULT_MODEL]


def test_a_full_install_still_uses_cached_non_gguf_rows():
    """Control for the gate above."""
    out = _run(
        "scenario({ chatOnly: false,"
        " modelRepos: [{ repo_id: 'org/st', load_id: 'org/st', size_bytes: 1 }] })"
    )
    assert _loaded_paths(out) == ["org/st"]


def test_a_provisional_platform_does_not_hide_cached_non_gguf_rows():
    """Before the probe lands the chatOnly flag is a browser guess."""
    out = _run(
        "scenario({ chatOnly: true, platformFetched: false,"
        " modelRepos: [{ repo_id: 'org/st', load_id: 'org/st', size_bytes: 1 }] })"
    )
    assert _loaded_paths(out) == ["org/st"]


def test_a_cached_adapter_repo_is_never_auto_loaded():
    """A cached LoRA has no weights of its own: /load fetches the base from the
    Hub when it is not cached, which is why the local twin is dropped too.
    Adapters are tiny, so the size order puts one first."""
    out = _run(
        "scenario({ modelRepos: [{ repo_id: 'org/lora', load_id: 'org/lora',"
        " size_bytes: 40000000, model_format: 'adapter',"
        " capabilities: { can_chat: true } }] })"
    )

    assert "org/lora" not in _loaded_paths(out)
    assert _loaded_paths(out) == [DEFAULT_MODEL]


def test_a_cached_safetensors_repo_is_still_auto_loaded():
    """Control for the gate above: only the adapter format is dropped."""
    out = _run(
        "scenario({ modelRepos: [{ repo_id: 'org/base', load_id: 'org/base',"
        " size_bytes: 40000000, model_format: 'safetensors',"
        " capabilities: { can_chat: true } }] })"
    )

    assert _loaded_paths(out) == ["org/base"]
    assert _downloads_started(out) == []


@pytest.mark.parametrize("broken", ["ggufRepos", "modelRepos"])
def test_an_hf_cache_row_is_used_when_the_cached_lookup_fails(broken):
    """/api/hub/local also reports hf_cache rows, which autoload normally skips
    as duplicates. When a cached list fails they are the only evidence left, and
    dropping them ended the send with no model at all, since the gap also blocks
    the default."""
    row = "{ ...LOCAL_GGUF, source: 'hf_cache' }"
    out = _run(f"scenario({{ {broken}: 'throw', localModels: [{row}] }})")

    assert _loaded_paths(out) == [LOCAL_GGUF_PATH]
    assert _downloads_started(out) == []


def test_an_hf_cache_row_stays_excluded_when_both_lookups_answer():
    """Control: with the cached lists healthy the row is a duplicate of what they
    already report, so autoload keeps skipping it."""
    row = "{ ...LOCAL_GGUF, source: 'hf_cache' }"
    out = _run(f"scenario({{ localModels: [{row}] }})")

    assert LOCAL_GGUF_PATH not in _loaded_paths(out)
    # Nothing loadable on device, so the default download is the correct answer.
    assert _loaded_paths(out) == [DEFAULT_MODEL]


def test_an_admitted_hf_cache_row_is_not_attempted_twice():
    """The surviving cached list may name the same model as the admitted
    hf_cache row, so the load-target dedupe has to collapse them or the pair
    spends two of the three attempts on one model."""
    row = (
        "{ ...LOCAL_GGUF, source: 'hf_cache', load_id: 'org/dup', path: 'org/dup', id: 'org/dup' }"
    )
    out = _run(
        "scenario({ ggufRepos: 'throw',"
        " modelRepos: [{ repo_id: 'org/dup', load_id: 'org/dup', size_bytes: 1 }],"
        f" localModels: [{row}] }})"
    )

    assert _loaded_paths(out).count("org/dup") == 1
    assert _downloads_started(out) == []


def test_a_safetensors_twin_survives_a_gguf_row_with_no_loadable_quant():
    """Dedupe keeps the GGUF row because the backend probes GGUF first, but when
    that repo resolves no quant -- every variant big-endian, say -- the twin was
    already discarded and a full inventory fell through to the default."""
    out = _run(
        "scenario({ ggufRepos: [{ ...GEMMA, repo_id: 'org/both', load_id: 'org/both' }],"
        " variants: { 'org/both': [{ quant: 'Q4_K_M',"
        " filename: 'gemma-4-26B-A4B-it-BE.Q4_K_M.gguf', downloaded: true,"
        " size_bytes: 900000000 }] },"
        " modelRepos: [{ repo_id: 'org/both', load_id: 'org/both', size_bytes: 2000000000 }] })"
    )

    assert _loaded_paths(out) == ["org/both"]
    assert _downloads_started(out) == []


def test_a_legacy_gguf_row_without_model_format_loads_as_gguf():
    """model_format is optional, so an older backend omits it on a direct .gguf
    row. The source builder did not fall back to the suffix, so the row became a
    Transformers source: /load got the safetensors context length instead of 0
    and the remembered kind was wrong."""
    row = "{ ...LOCAL_GGUF, model_format: undefined }"
    out = _run(f"scenario({{ localModels: [{row}] }})")

    assert _loaded_paths(out) == [LOCAL_GGUF_PATH]
    remembered = [e for e in out["events"] if e["kind"] == "recordLastLocal"]
    assert remembered and remembered[0]["modelKind"] == "gguf", remembered


@pytest.mark.parametrize("fmt", ["'unknown'", "undefined"])
def test_an_unclassified_local_row_is_never_auto_loaded(fmt):
    """The gate was a denylist, so a row the backend could not classify passed
    it: "unknown" is what the backend sends when it cannot tell, and an older one
    omits the field. Either way the row may be the pickle checkpoint the
    exclusions exist to keep out, and a directory gives no suffix to tell them
    apart, so this fails closed."""
    row = f"{{ ...LOCAL_GGUF, id: 'x', load_id: '/models/x', path: '/models/x', model_format: {fmt} }}"
    out = _run(f"scenario({{ localModels: [{row}] }})")

    assert "/models/x" not in _loaded_paths(out)
    assert _loaded_paths(out) == [DEFAULT_MODEL]


_EXPIRE_CLI_LOAD_WAIT = """
const realNow = Date.now.bind(Date);
let nowCalls = 0;
Date.now = () => realNow() + (nowCalls++ === 0 ? 0 : 600_001);
"""


def test_send_retries_status_then_waits_for_and_adopts_the_cli_model():
    """Send retries status, waits for the CLI load, and adopts its model."""
    out = _run(
        "scenario({ statusFailures: 1, serverLoading: ['org/slow-model-GGUF'],"
        " serverLoadingClearsAfter: 2,"
        " ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS } })"
    )

    adopted = [e["id"] for e in out["events"] if e["kind"] == "adoptServerModel"]
    assert adopted == ["org/slow-model-GGUF"]
    assert _loaded_paths(out) == []
    assert _downloads_started(out) == []
    assert out["result"]["loaded"] is True
    # Announced once, not once per poll.
    assert [t["msg"] for t in _toasts(out, "toast.info")] == [
        "Waiting for model to finish loading…"
    ]


def test_a_load_still_in_flight_at_the_cap_refuses_instead_of_auto_loading():
    """The wait cap must not release auto-load over a running CLI load."""
    out = _run(
        "scenario({ serverLoading: ['org/slow-model-GGUF'],"
        " ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS } })",
        prelude = _EXPIRE_CLI_LOAD_WAIT,
    )

    assert _loaded_paths(out) == []
    assert _downloads_started(out) == []
    assert out["result"]["loaded"] is False
    assert out["result"]["loadFailureReported"] is True
    assert [t["msg"] for t in _toasts(out, "toast.error")] == ["A model is still loading"]


def test_an_idle_server_still_auto_loads():
    """An idle server still reaches automatic loading."""
    out = _run("scenario({ ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS } })")

    assert _loaded_paths(out) == [GEMMA_REPO]
    assert out["result"]["loaded"] is True
    assert _toasts(out, "toast.info") == []


def test_a_status_endpoint_that_never_answers_refuses_rather_than_guessing():
    """An unavailable status endpoint must not be treated as idle."""
    out = _run(
        "scenario({ statusFailures: 99, ggufRepos: [GEMMA],"
        " variants: { [GEMMA.repo_id]: GEMMA_VARIANTS } })"
    )

    assert _loaded_paths(out) == []
    assert out["result"]["loaded"] is False
    assert out["result"]["loadFailureReported"] is True
    assert [t["msg"] for t in _toasts(out, "toast.error")] == ["Could not reach the model server"]


# A queued turn whose thread carries no model resolves one of its own, and the visible
# tab may be on a provider model meanwhile. That resolver reads /status directly, so it
# needs the same in-flight-load gate the sweep above has.


def test_a_queued_turn_binds_the_incoming_model_not_the_one_being_replaced():
    """During a replacement the status names the resident model and the incoming one at
    once, and only the incoming one is what this turn will actually run against."""
    out = _run(
        "scenario({ visibleCheckpoint: EXTERNAL, serverResident: 'org/outgoing-GGUF',"
        " serverLoading: ['org/slow-model-GGUF'], serverLoadingClearsAfter: 2,"
        " ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS } })",
        queued = True,
    )

    assert out["result"]["loaded"] is True
    assert out["result"]["modelRuntime"]["checkpoint"] == "org/slow-model-GGUF"
    assert _loaded_paths(out) == []
    assert _downloads_started(out) == []


def test_a_queued_turn_refuses_rather_than_auto_loading_over_a_cli_load():
    """The wait cap must not release the queued sweep over a running CLI load either."""
    out = _run(
        "scenario({ visibleCheckpoint: EXTERNAL, serverLoading: ['org/slow-model-GGUF'],"
        " ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS } })",
        prelude = _EXPIRE_CLI_LOAD_WAIT,
        queued = True,
    )

    assert _loaded_paths(out) == []
    assert _downloads_started(out) == []
    assert out["result"]["loaded"] is False
    assert out["result"]["loadFailureReported"] is True
    assert out["result"]["modelRuntime"] is None
    assert [t["msg"] for t in _toasts(out, "toast.error")] == ["A model is still loading"]


def test_a_queued_turn_still_binds_a_resident_model_on_an_idle_server():
    out = _run(
        "scenario({ visibleCheckpoint: EXTERNAL, serverResident: 'org/resident-GGUF',"
        " ggufRepos: [GEMMA], variants: { [GEMMA.repo_id]: GEMMA_VARIANTS } })",
        queued = True,
    )

    assert out["result"]["loaded"] is True
    assert out["result"]["modelRuntime"]["checkpoint"] == "org/resident-GGUF"
    assert _loaded_paths(out) == []
    assert _toasts(out, "toast.info") == []


def test_a_queued_turn_on_an_empty_idle_server_still_auto_loads():
    out = _run(
        "scenario({ visibleCheckpoint: EXTERNAL, ggufRepos: [GEMMA],"
        " variants: { [GEMMA.repo_id]: GEMMA_VARIANTS } })",
        queued = True,
    )

    assert _loaded_paths(out) == [GEMMA_REPO]
    assert out["result"]["loaded"] is True


def test_an_unpinned_mlx_candidate_is_loaded_at_the_window_the_backend_resolved():
    """The auto-load runs unwatched, so its request is what an MLX model is left serving;
    the app default there loaded every model at 4096."""
    out = _run("scenario({ modelRepos: [QWEN], platform: MAC, load: SERVED(262144) })")
    [load] = [e for e in out["events"] if e["kind"] == "loadModel"]
    assert load["model_path"] == "unsloth/Qwen3.5-4B"
    # The sentinel that hands sizing to the backend, not a length chosen here.
    assert load["max_seq_length"] == 0
    # And what comes back describes the model rather than the request.
    assert out["store"]["maxSeqLength"] == 262144
    assert out["store"]["maxTokens"] == 262144
    # Nothing was pinned, so nothing is remembered as pinned.
    assert out["store"]["customContextLength"] is None
    assert out["store"]["loadedCustomContextLength"] is None


def test_a_pinned_mlx_candidate_is_loaded_at_its_pin_and_keeps_it():
    """The rules are proved apart; this is the only proof the auto-load threads them: the
    record's pin into the request, and the retained pin back. Dropping either leaves a
    pinned model asking for 0, or forgetting its pin on reload."""
    out = _run(
        "scenario({ modelRepos: [QWEN], platform: MAC,"
        " config: { customContextLength: 8192 }, load: SERVED(8192) })"
    )
    [load] = [e for e in out["events"] if e["kind"] == "loadModel"]
    assert load["max_seq_length"] == 8192
    assert out["store"]["customContextLength"] == 8192
    assert out["store"]["loadedCustomContextLength"] == 8192
    assert out["store"]["maxSeqLength"] == 8192
    # A pre-move record holds the pin in the other field; the write-back moves it.
    legacy = _run(
        "scenario({ modelRepos: [QWEN], platform: MAC,"
        " config: { maxSeqLength: 8192 }, load: SERVED(8192) })"
    )
    assert [e for e in legacy["events"] if e["kind"] == "loadModel"][0]["max_seq_length"] == 8192
    assert legacy["store"]["customContextLength"] == 8192
