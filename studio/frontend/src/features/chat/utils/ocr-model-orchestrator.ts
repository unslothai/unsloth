// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";
import {
  getDocumentSupport,
  getInferenceStatus,
  invalidateDocumentSupportCache,
  loadModel,
  unloadModel,
  validateModel,
} from "../api/chat-api";
import {
  type DocExtractSettings,
  type OcrPhase,
  type ReasoningStyle,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";
import type {
  InferenceStatusResponse,
  LoadModelRequest,
  LoadModelResponse,
} from "../types/api";
import type { InferenceParams } from "../types/runtime";
import {
  type OcrModelTarget,
  resolveOcrModelTarget,
} from "./ocr-model-presets";
import {
  acquireTemporaryOcrModelLease,
  setTemporaryOcrModelBusy,
  type TemporaryOcrModelLease,
  waitForTemporaryOcrModelIdle,
} from "./ocr-model-lock";

export type { OcrPhase };

export interface ChatModelSnapshot {
  checkpoint: string;
  ggufVariant: string | null;
  trustRemoteCode: boolean;
  maxSeqLength: number;
  loadIn4Bit: boolean;
  isLora: boolean;
  ggufContextLength: number | null;
  ggufMaxContextLength: number | null;
  ggufNativeContextLength: number | null;
  kvCacheDtype: string | null;
  loadedKvCacheDtype: string | null;
  speculativeType: string | null;
  loadedSpeculativeType: string | null;
  customContextLength: number | null;
  chatTemplateOverride: string | null;
  defaultChatTemplate: string | null;
  modelRequiresTrustRemoteCode: boolean;
  supportsReasoning: boolean;
  reasoningAlwaysOn: boolean;
  reasoningEnabled: boolean;
  reasoningStyle: ReasoningStyle;
  supportsPreserveThinking: boolean;
  supportsTools: boolean;
  toolsEnabled: boolean;
  codeToolsEnabled: boolean;
}

export interface RunWithTemporaryOcrModelArgs<T> {
  settings: DocExtractSettings;
  signal?: AbortSignal;
  run: () => Promise<T>;
}

function needsTemporaryOcrWorker(settings: DocExtractSettings): boolean {
  return (
    resolveOcrModelTarget(settings) !== null &&
    settings.enabled &&
    (settings.useVlmOcr || settings.describeImages)
  );
}

function clearStaleOcrErrorPhase(): void {
  if (useChatRuntimeStore.getState().ocrPhase === "error") {
    setOcrPhase("idle");
  }
}

/**
 * Run `args.run()` against the OCR model selected in `args.settings`.
 *
 * Lifecycle, in order:
 *   1. Resolve the target — if the user picked "default"/"none" or extraction
 *      is disabled, run the inner function directly with no model swap.
 *   2. Validate the OCR model. If validation fails (or trust_remote_code is
 *      required and the user has it disabled), reject before unloading.
 *   3. If a chat model is loaded and not already the OCR target, unload it.
 *   4. Load the OCR model.
 *   5. Run the inner function (extraction).
 *   6. In `finally`, restore the snapshot — but never overwrite a manual
 *      mid-run model swap. Reconcile the store from `getInferenceStatus()`
 *      if the active model changed.
 *
 * Concurrent calls are serialized through a module-level promise queue so
 * two simultaneous uploads never fight over the global active model.
 */
export async function runWithTemporaryOcrModel<T>(
  args: RunWithTemporaryOcrModelArgs<T>,
): Promise<T> {
  if (!needsTemporaryOcrWorker(args.settings)) {
    return runPassThrough(args);
  }

  pendingSwapRuns += 1;
  const runExclusive = async () => {
    await waitForPassThroughIdle();
    return runUnlocked(args);
  };
  const next = queue.then(
    runExclusive,
    runExclusive,
  );
  queue = next.then(
    () => undefined,
    () => undefined,
  );
  try {
    return await next;
  } finally {
    pendingSwapRuns -= 1;
  }
}

/** Test helper. Resets the module-level queue and loading gate. */
export function resetOcrModelQueueForTests(): void {
  queue = Promise.resolve();
  pendingSwapRuns = 0;
  activePassThroughRuns = 0;
  passThroughIdleWaiters = [];
  setModelLoading(false);
  setTemporaryOcrModelBusy(false);
}

let queue: Promise<void> = Promise.resolve();
let pendingSwapRuns = 0;
let activePassThroughRuns = 0;
let passThroughIdleWaiters: Array<() => void> = [];

interface OcrIdentity {
  checkpoint: string;
  ggufVariant: string | null;
}

function setOcrPhase(phase: OcrPhase): void {
  useChatRuntimeStore.getState().setOcrPhase(phase);
}

function setModelLoading(loading: boolean): void {
  useChatRuntimeStore.getState().setModelLoading(loading);
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException("Aborted", "AbortError");
  }
}

async function runPassThrough<T>({
  signal,
  run,
}: RunWithTemporaryOcrModelArgs<T>): Promise<T> {
  while (pendingSwapRuns > 0) {
    await queue;
  }
  await waitForTemporaryOcrModelIdle(signal);
  clearStaleOcrErrorPhase();
  activePassThroughRuns += 1;
  try {
    return await run();
  } finally {
    activePassThroughRuns -= 1;
    if (activePassThroughRuns === 0) {
      const waiters = passThroughIdleWaiters;
      passThroughIdleWaiters = [];
      waiters.forEach((resolve) => resolve());
    }
  }
}

function waitForPassThroughIdle(): Promise<void> {
  if (activePassThroughRuns === 0) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    passThroughIdleWaiters.push(resolve);
  });
}

function captureSnapshot(): ChatModelSnapshot {
  const state = useChatRuntimeStore.getState();
  const activeModel = state.models.find(
    (model) => model.id === state.params.checkpoint,
  );
  const activeLora = state.loras.find(
    (lora) => lora.id === state.params.checkpoint,
  );
  const activeIsLora =
    activeModel?.isLora ?? (activeLora?.exportType === "lora");
  return {
    checkpoint: state.params.checkpoint,
    ggufVariant: state.activeGgufVariant,
    trustRemoteCode: state.params.trustRemoteCode ?? false,
    maxSeqLength: state.params.maxSeqLength,
    loadIn4Bit: state.params.loadIn4Bit ?? true,
    isLora: activeIsLora,
    ggufContextLength: state.ggufContextLength,
    ggufMaxContextLength: state.ggufMaxContextLength,
    ggufNativeContextLength: state.ggufNativeContextLength,
    kvCacheDtype: state.kvCacheDtype,
    loadedKvCacheDtype: state.loadedKvCacheDtype,
    speculativeType: state.speculativeType,
    loadedSpeculativeType: state.loadedSpeculativeType,
    customContextLength: state.customContextLength,
    chatTemplateOverride: state.chatTemplateOverride,
    defaultChatTemplate: state.defaultChatTemplate,
    modelRequiresTrustRemoteCode: state.modelRequiresTrustRemoteCode,
    supportsReasoning: state.supportsReasoning,
    reasoningAlwaysOn: state.reasoningAlwaysOn,
    reasoningEnabled: state.reasoningEnabled,
    reasoningStyle: state.reasoningStyle,
    supportsPreserveThinking: state.supportsPreserveThinking,
    supportsTools: state.supportsTools,
    toolsEnabled: state.toolsEnabled,
    codeToolsEnabled: state.codeToolsEnabled,
  };
}

function sameIdentity(a: OcrIdentity, b: OcrIdentity): boolean {
  return a.checkpoint === b.checkpoint && a.ggufVariant === b.ggufVariant;
}

function identityFromStore(): OcrIdentity {
  const state = useChatRuntimeStore.getState();
  return {
    checkpoint: state.params.checkpoint,
    ggufVariant: state.activeGgufVariant,
  };
}

function buildOcrLoadPayload(
  target: OcrModelTarget,
  snapshot: ChatModelSnapshot,
): LoadModelRequest {
  const hfToken = useChatRuntimeStore.getState().hfToken;
  return {
    model_path: target.modelId,
    hf_token: hfToken || null,
    max_seq_length: target.defaultMaxSeqLength,
    load_in_4bit: snapshot.loadIn4Bit,
    is_lora: false,
    gguf_variant: target.ggufVariant,
    trust_remote_code: snapshot.trustRemoteCode,
  };
}

function buildRestorePayload(snapshot: ChatModelSnapshot): LoadModelRequest {
  const hfToken = useChatRuntimeStore.getState().hfToken;
  const isGguf =
    snapshot.ggufVariant !== null ||
    snapshot.checkpoint.toLowerCase().endsWith(".gguf");
  const effectiveMaxSeqLength =
    snapshot.customContextLength ??
    (isGguf ? (snapshot.ggufContextLength ?? 0) : snapshot.maxSeqLength);
  return {
    model_path: snapshot.checkpoint,
    hf_token: hfToken || null,
    max_seq_length: effectiveMaxSeqLength,
    load_in_4bit: snapshot.loadIn4Bit,
    is_lora: snapshot.isLora,
    gguf_variant: snapshot.ggufVariant,
    trust_remote_code: snapshot.trustRemoteCode,
    chat_template_override: snapshot.chatTemplateOverride,
    cache_type_kv: snapshot.kvCacheDtype,
    speculative_type: snapshot.speculativeType,
  };
}

function toFiniteNumber(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  return value;
}

function normalizeSpeculativeType(v: string | null | undefined): string | null {
  if (v == null) return null;
  if (v === "default" || v === "off") return v;
  return "default";
}

function mergeRecommendedInference(
  current: InferenceParams,
  response: LoadModelResponse | InferenceStatusResponse,
  modelId: string,
): InferenceParams {
  const inference = response.inference;
  const defaultMaxTokens = response.is_gguf
    ? (response.context_length ?? 131072)
    : 4096;
  return {
    ...current,
    checkpoint: modelId,
    maxTokens: defaultMaxTokens,
    temperature:
      toFiniteNumber(inference?.temperature) ?? current.temperature,
    topP: toFiniteNumber(inference?.top_p) ?? current.topP,
    topK: toFiniteNumber(inference?.top_k) ?? current.topK,
    minP: toFiniteNumber(inference?.min_p) ?? current.minP,
    presencePenalty:
      toFiniteNumber(inference?.presence_penalty) ?? current.presencePenalty,
    trustRemoteCode:
      typeof inference?.trust_remote_code === "boolean"
        ? inference.trust_remote_code
        : current.trustRemoteCode,
  };
}

function defaultReasoningEnabledForModel(
  modelId: string,
  supportsReasoning: boolean,
): boolean {
  if (!supportsReasoning) return true;
  const mid = modelId.toLowerCase();
  if (mid.includes("qwen3.5") || mid.includes("qwen3.6")) {
    const sizeMatch = mid.match(/(\d+\.?\d*)\s*b/);
    if (sizeMatch && parseFloat(sizeMatch[1]) < 9) {
      return false;
    }
  }
  return true;
}

function applyLoadedModelToStore(
  modelId: string,
  ggufVariant: string | null,
  loaded: LoadModelResponse,
  preserve?: ChatModelSnapshot,
): void {
  const store = useChatRuntimeStore.getState();
  store.setCheckpoint(modelId, loaded.is_gguf ? ggufVariant : null);

  const paramsState = useChatRuntimeStore.getState();
  paramsState.setParams(
    mergeRecommendedInference(paramsState.params, loaded, modelId),
  );

  const supportsReasoning =
    loaded.supports_reasoning ?? preserve?.supportsReasoning ?? false;
  const reasoningAlwaysOn =
    loaded.reasoning_always_on ?? preserve?.reasoningAlwaysOn ?? false;
  const reasoningDefault = defaultReasoningEnabledForModel(
    modelId,
    supportsReasoning,
  );
  const supportsTools = loaded.supports_tools ?? preserve?.supportsTools ?? false;
  const loadedSpec =
    normalizeSpeculativeType(loaded.speculative_type) ??
    preserve?.loadedSpeculativeType ??
    preserve?.speculativeType ??
    null;
  const loadedKv = loaded.cache_type_kv ?? null;

  useChatRuntimeStore.setState({
    ggufContextLength: loaded.is_gguf
      ? (loaded.context_length ?? preserve?.ggufContextLength ?? 131072)
      : null,
    ggufMaxContextLength: loaded.is_gguf
      ? (loaded.max_context_length ?? preserve?.ggufMaxContextLength ?? null)
      : null,
    ggufNativeContextLength: loaded.is_gguf
      ? (loaded.native_context_length ??
        preserve?.ggufNativeContextLength ??
        null)
      : null,
    modelRequiresTrustRemoteCode:
      loaded.requires_trust_remote_code ??
      preserve?.modelRequiresTrustRemoteCode ??
      false,
    supportsReasoning,
    reasoningAlwaysOn,
    reasoningEnabled: reasoningAlwaysOn
      ? true
      : supportsReasoning
        ? (preserve?.reasoningEnabled ?? reasoningDefault)
        : true,
    reasoningStyle:
      loaded.reasoning_style ?? preserve?.reasoningStyle ?? "enable_thinking",
    supportsPreserveThinking:
      loaded.supports_preserve_thinking ??
      preserve?.supportsPreserveThinking ??
      false,
    supportsTools,
    toolsEnabled: supportsTools ? (preserve?.toolsEnabled ?? true) : false,
    codeToolsEnabled: supportsTools
      ? (preserve?.codeToolsEnabled ?? true)
      : false,
    kvCacheDtype: loadedKv,
    loadedKvCacheDtype: loadedKv,
    speculativeType: loadedSpec,
    loadedSpeculativeType: loadedSpec,
    customContextLength: null,
    defaultChatTemplate: loaded.chat_template ?? preserve?.defaultChatTemplate ?? null,
    chatTemplateOverride: null,
  });
}

function applyStatusToStore(status: InferenceStatusResponse): void {
  const store = useChatRuntimeStore.getState();
  if (!status.active_model) {
    store.clearCheckpoint();
    return;
  }

  store.setCheckpoint(status.active_model, status.gguf_variant ?? null);
  if (status.inference) {
    const paramsState = useChatRuntimeStore.getState();
    paramsState.setParams(
      mergeRecommendedInference(
        paramsState.params,
        status,
        status.active_model,
      ),
    );
  }

  const current = useChatRuntimeStore.getState();
  const supportsReasoning = status.supports_reasoning ?? false;
  const reasoningAlwaysOn = status.reasoning_always_on ?? false;
  const supportsTools = status.supports_tools ?? false;
  const currentSpecType = normalizeSpeculativeType(status.speculative_type);
  const loadedKv = status.cache_type_kv ?? null;
  useChatRuntimeStore.setState({
    supportsReasoning,
    reasoningAlwaysOn,
    reasoningStyle: status.reasoning_style ?? "enable_thinking",
    supportsPreserveThinking: status.supports_preserve_thinking ?? false,
    supportsTools,
    reasoningEnabled: reasoningAlwaysOn
      ? true
      : supportsReasoning
        ? current.reasoningEnabled
        : true,
    toolsEnabled: supportsTools ? current.toolsEnabled : false,
    codeToolsEnabled: supportsTools ? current.codeToolsEnabled : false,
    kvCacheDtype: loadedKv,
    loadedKvCacheDtype: loadedKv,
    ggufContextLength: status.is_gguf ? (status.context_length ?? null) : null,
    ggufMaxContextLength: status.is_gguf
      ? (status.max_context_length ?? null)
      : null,
    ggufNativeContextLength: status.is_gguf
      ? (status.native_context_length ?? null)
      : null,
    modelRequiresTrustRemoteCode: status.requires_trust_remote_code ?? false,
    speculativeType: currentSpecType,
    loadedSpeculativeType: currentSpecType,
  });
}

async function reconcileStoreFromStatus(): Promise<void> {
  try {
    const status = await getInferenceStatus();
    applyStatusToStore(status);
  } catch {
    // Best-effort reconciliation; never fabricate state on failure.
  }
}

function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  return "Unknown error";
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

async function runUnlocked<T>({
  settings,
  signal,
  run,
}: RunWithTemporaryOcrModelArgs<T>): Promise<T> {
  // A previous run may have left ocrPhase="error" on its way out. Clear it
  // here so the UI banner from that earlier failure doesn't bleed into the
  // start of this run.
  if (useChatRuntimeStore.getState().ocrPhase === "error") {
    setOcrPhase("idle");
  }
  const target = resolveOcrModelTarget(settings);
  const needsWorker =
    target !== null &&
    settings.enabled &&
    (settings.useVlmOcr || settings.describeImages);

  if (!needsWorker || target === null) {
    return run();
  }

  const ocrIdentity: OcrIdentity = {
    checkpoint: target.modelId,
    ggufVariant: target.ggufVariant,
  };

  setOcrPhase("validating");
  setModelLoading(true);
  let lease: TemporaryOcrModelLease | null = null;
  let snapshot: ChatModelSnapshot | null = null;
  let alreadyActive = false;
  let didSwap = false;
  let previousUnloadRequested = false;

  try {
    throwIfAborted(signal);
    lease = await acquireTemporaryOcrModelLease(signal);
    snapshot = captureSnapshot();
    alreadyActive =
      snapshot.checkpoint.length > 0 &&
      sameIdentity(
        { checkpoint: snapshot.checkpoint, ggufVariant: snapshot.ggufVariant },
        ocrIdentity,
      );
    throwIfAborted(signal);

    const validation = await validateModel(
      buildOcrLoadPayload(target, snapshot),
      signal,
    );
    if (!validation.valid) {
      throw new Error(
        validation.message || `${target.label} failed validation.`,
      );
    }
    if (validation.is_vision === false) {
      throw new Error(`${target.label} is not vision-capable.`);
    }
    if (validation.requires_trust_remote_code && !snapshot.trustRemoteCode) {
      throw new Error(
        `${target.label} requires "Enable custom code". Turn it on under ` +
          "Inference settings before scanning.",
      );
    }

    if (!alreadyActive) {
      lease.assertActive();
      if (snapshot.checkpoint) {
        setOcrPhase("unloading");
        throwIfAborted(signal);
        lease.assertActive();
        previousUnloadRequested = true;
        await unloadModel({ model_path: snapshot.checkpoint }, signal);
        useChatRuntimeStore.getState().clearCheckpoint();
      }

      setOcrPhase("loading_ocr");
      throwIfAborted(signal);
      lease.assertActive();
      const loaded = await loadModel(
        buildOcrLoadPayload(target, snapshot),
        signal,
      );
      lease.assertActive();
      if (loaded.is_vision === false) {
        throw new Error(
          `Loaded ${target.label} did not report vision support.`,
        );
      }
      applyLoadedModelToStore(target.modelId, target.ggufVariant, loaded);
      invalidateDocumentSupportCache();
      // Bounded probe: wait until the server-side document-support endpoint
      // reports the OCR model as the active VLM, so any UI consumer that
      // re-reads support during extraction sees the up-to-date capability.
      // Times out silently — extraction itself uses runtime detect_loaded_vlm()
      // and is unaffected by stale cache.
      await waitForDocumentSupportVision(ocrIdentity, signal);
      didSwap = true;
    }

    setOcrPhase("extracting");
    lease.assertActive();
    return await run();
  } catch (err) {
    const phaseAtError = useChatRuntimeStore.getState().ocrPhase;
    setOcrPhase("error");
    if (!isAbortError(err)) {
      const failureSnapshot = snapshot ?? captureSnapshot();
      const { title, description } = describeFailure(
        phaseAtError,
        target,
        failureSnapshot,
        err,
      );
      toast.error(title, { description });
    }
    throw err;
  } finally {
    try {
      if (lease && !lease.isActive()) {
        toast.info(
          "Skipped restoring previous chat model — OCR model lock was lost during extraction.",
        );
        await reconcileStoreFromStatus();
      } else if (snapshot && didSwap) {
        setOcrPhase("restoring");
        await restoreSnapshotOrReconcile(snapshot, ocrIdentity);
      } else if (snapshot && previousUnloadRequested && snapshot.checkpoint) {
        await restoreUnloadedSnapshot(snapshot, ocrIdentity);
      }
    } finally {
      invalidateDocumentSupportCache();
      setOcrPhase("idle");
      setModelLoading(false);
      lease?.release();
    }
  }
}

async function restoreUnloadedSnapshot(
  snapshot: ChatModelSnapshot,
  attemptedOcrIdentity?: OcrIdentity,
): Promise<void> {
  setOcrPhase("restoring");
  const currentInUi = identityFromStore();
  let serverActive: OcrIdentity | null | undefined;
  try {
    const status = await getInferenceStatus();
    serverActive = status.active_model
      ? {
          checkpoint: status.active_model,
          ggufVariant: status.gguf_variant ?? null,
        }
      : null;
  } catch {
    serverActive = undefined;
  }

  const snapshotIdentity = {
    checkpoint: snapshot.checkpoint,
    ggufVariant: snapshot.ggufVariant,
  };
  const uiStillOwned =
    currentInUi.checkpoint.length === 0 ||
    sameIdentity(currentInUi, snapshotIdentity) ||
    (attemptedOcrIdentity !== undefined &&
      sameIdentity(currentInUi, attemptedOcrIdentity));
  const serverStillOwned =
    serverActive === undefined ||
    serverActive === null ||
    sameIdentity(serverActive, snapshotIdentity) ||
    (attemptedOcrIdentity !== undefined &&
      sameIdentity(serverActive, attemptedOcrIdentity));

  if (!uiStillOwned || !serverStillOwned) {
    toast.info(
      "Skipped restoring previous chat model — active model changed during extraction.",
    );
    await reconcileStoreFromStatus();
    return;
  }

  try {
    const restored = await loadModel(buildRestorePayload(snapshot));
    applyLoadedModelToStore(
      snapshot.checkpoint,
      snapshot.ggufVariant,
      restored,
      snapshot,
    );
  } catch (err) {
    toast.warning(`Could not restore ${snapshot.checkpoint || "chat model"}.`, {
      description: errorMessage(err),
      duration: Number.POSITIVE_INFINITY,
      action: snapshot.checkpoint
        ? {
            label:
              snapshot.checkpoint.length > 28
                ? `Reload ${snapshot.checkpoint.slice(0, 25)}…`
                : `Reload ${snapshot.checkpoint}`,
            onClick: () => {
              void enqueueRestoreRetry(snapshot);
            },
          }
        : undefined,
    });
    await reconcileStoreFromStatus();
  } finally {
    invalidateDocumentSupportCache();
    setOcrPhase("idle");
    setModelLoading(false);
  }
}

async function restoreSnapshotOrReconcile(
  snapshot: ChatModelSnapshot,
  ocrIdentity: OcrIdentity,
): Promise<void> {
  // If the user manually swapped models mid-run, never overwrite — reconcile.
  const currentInUi = identityFromStore();
  let serverActive: string | null = null;
  let serverVariant: string | null = null;
  try {
    const status = await getInferenceStatus();
    serverActive = status.active_model ?? null;
    serverVariant = status.gguf_variant ?? null;
  } catch {
    // Fall back to UI identity if status fetch fails.
  }

  const userChangedModelMidRun =
    !sameIdentity(currentInUi, ocrIdentity) ||
    (serverActive !== null &&
      !sameIdentity(
        { checkpoint: serverActive, ggufVariant: serverVariant },
        ocrIdentity,
      ));

  if (userChangedModelMidRun) {
    toast.info(
      "Skipped restoring previous chat model — active model changed during extraction.",
    );
    await reconcileStoreFromStatus();
    return;
  }

  try {
    if (snapshot.checkpoint) {
      const restored = await loadModel(buildRestorePayload(snapshot));
      applyLoadedModelToStore(
        snapshot.checkpoint,
        snapshot.ggufVariant,
        restored,
        snapshot,
      );
    } else {
      // No prior chat model — drop the OCR model so we end in a clean state.
      await unloadModel({ model_path: ocrIdentity.checkpoint });
      useChatRuntimeStore.getState().clearCheckpoint();
    }
  } catch (err) {
    const labelText = snapshot.checkpoint
      ? snapshot.checkpoint.length > 28
        ? `Reload ${snapshot.checkpoint.slice(0, 25)}…`
        : `Reload ${snapshot.checkpoint}`
      : null;
    toast.warning(`Could not restore ${snapshot.checkpoint || "chat model"}.`, {
      description: errorMessage(err),
      // Sticky toast — clears on user dismiss, retry, or route navigation.
      duration: Number.POSITIVE_INFINITY,
      action:
        snapshot.checkpoint && labelText
          ? {
              label: labelText,
              onClick: () => {
                void enqueueRestoreRetry(snapshot);
              },
            }
          : undefined,
    });
    await reconcileStoreFromStatus();
  }
}

// UI-accuracy poll only; extract correctness uses runtime detect_loaded_vlm()
// regardless. Capped low because this runs inside the orchestrator queue —
// every extra second blocks subsequent uploads from starting.
const VISION_PROBE_MAX_MS = 2000;
const VISION_PROBE_INTERVAL_MS = 500;

/**
 * Maps a failed OCR phase to a user-facing toast title + description so the
 * surface error message reflects which step actually broke (validation vs
 * unload vs load vs restore).
 */
function describeFailure(
  phase: OcrPhase,
  target: OcrModelTarget,
  snapshot: ChatModelSnapshot,
  err: unknown,
): { title: string; description: string } {
  const reason = errorMessage(err);
  const chatLabel = snapshot.checkpoint || "your chat model";
  switch (phase) {
    case "validating":
      return {
        title: "OCR model failed validation",
        description: `${target.label}: ${reason}. Chat model not unloaded.`,
      };
    case "unloading":
      return {
        title: "Could not unload current chat model",
        description: reason,
      };
    case "loading_ocr":
      return {
        title: `Could not load ${target.label}`,
        description: reason,
      };
    case "extracting":
      return {
        title: "Document extraction failed",
        description: reason,
      };
    case "restoring":
      return {
        title: `Could not restore ${chatLabel}`,
        description: reason,
      };
    default:
      return { title: "OCR run failed", description: reason };
  }
}

/**
 * Re-attempt loading the snapshot's chat model. Bound to the failed
 * orchestrator run's snapshot so the user can recover from a restore failure
 * via the toast action without re-running the divergence checks (which would
 * trip on the user's previous chat model still being absent server-side).
 *
 * The leading equality check short-circuits if a subsequent run already
 * restored the model.
 */
function enqueueRestoreRetry(snapshot: ChatModelSnapshot): Promise<void> {
  const restored = queue.then(
    () => retryRestoreSnapshot(snapshot),
    () => retryRestoreSnapshot(snapshot),
  );
  queue = restored.then(
    () => undefined,
    () => undefined,
  );
  return restored;
}

async function retryRestoreSnapshot(
  snapshot: ChatModelSnapshot,
): Promise<void> {
  if (!snapshot.checkpoint) return;
  const live = useChatRuntimeStore.getState();
  if (live.params.checkpoint === snapshot.checkpoint) {
    toast.info(`${snapshot.checkpoint} is already loaded.`);
    return;
  }
  try {
    setOcrPhase("restoring");
    setModelLoading(true);
    const restored = await loadModel(buildRestorePayload(snapshot));
    applyLoadedModelToStore(
      snapshot.checkpoint,
      snapshot.ggufVariant,
      restored,
      snapshot,
    );
    toast.success(`Reloaded ${snapshot.checkpoint}.`);
  } catch (retryErr) {
    toast.error(`Could not reload ${snapshot.checkpoint}.`, {
      description: errorMessage(retryErr),
    });
    await reconcileStoreFromStatus();
  } finally {
    invalidateDocumentSupportCache();
    setOcrPhase("idle");
    setModelLoading(false);
  }
}

/**
 * Bounded poll on the document-support endpoint after an OCR model load,
 * waiting until the server reports a vision-capable model. Bypasses the
 * 30 s `documentSupportCache` so UI consumers don't briefly observe the
 * pre-OCR vision state. Times out silently — the extract route uses runtime
 * `detect_loaded_vlm()` so correctness is unaffected.
 */
async function waitForDocumentSupportVision(
  expected: OcrIdentity,
  signal?: AbortSignal,
): Promise<void> {
  const deadline = Date.now() + VISION_PROBE_MAX_MS;
  while (Date.now() < deadline) {
    if (signal?.aborted) return;
    try {
      const support = await getDocumentSupport(signal);
      if (support.vlm?.is_vlm) {
        const reportedId = support.vlm.model_name ?? null;
        // Best signal: model_name matches the OCR id we just loaded.
        if (!reportedId || reportedId === expected.checkpoint) return;
      }
    } catch {
      // Network blip; loop until deadline.
    }
    await new Promise((resolve) =>
      setTimeout(resolve, VISION_PROBE_INTERVAL_MS),
    );
  }
}
