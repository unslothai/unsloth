// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { create } from "zustand";
import { invalidateDocumentSupportCache } from "../api/chat-api";
import {
  type ChatPresetSource,
  type Preset,
  getPresetSource,
} from "../presets/preset-policy";
import {
  type ChatLoraSummary,
  type ChatModelSummary,
  DEFAULT_INFERENCE_PARAMS,
  type InferenceParams,
} from "../types/runtime";
import { isExternalModelId, parseExternalModelId } from "../external-providers";
import { getExternalMaxOutputTokens } from "../provider-capabilities";
import { useExternalProvidersStore } from "./external-providers-store";
import {
  loadChatSettingsWithLegacyImport,
  savePersistedChatSettingsPatch,
} from "../utils/chat-settings-storage";

const HF_TOKEN_KEY = "unsloth_hf_token";
const INFERENCE_PARAMS_KEY = "unsloth_chat_inference_params";
const CHAT_ACTIVE_PRESET_KEY = "unsloth_chat_active_preset";
const CHAT_ACTIVE_PRESET_SOURCE_KEY = "unsloth_chat_active_preset_source";
const REASONING_EFFORT_KEY = "unsloth_reasoning_effort";
const PRESERVE_THINKING_KEY = "unsloth_preserve_thinking";
const DOC_EXTRACT_KEY = "unsloth_chat_doc_extract";
const DEFAULT_DOCUMENT_VISUAL_PAYLOADS = 3;
const DEFAULT_EXTRACT_CONCURRENCY = 2;
const MAX_EXTRACT_CONCURRENCY = 8;

export const CHAT_REASONING_ENABLED_KEY = "unsloth_chat_reasoning_enabled";
export const CHAT_TOOLS_ENABLED_KEY = "unsloth_chat_tools_enabled";
export const CHAT_CODE_TOOLS_ENABLED_KEY = "unsloth_chat_code_tools_enabled";
export const CHAT_IMAGE_TOOLS_ENABLED_KEY = "unsloth_chat_image_tools_enabled";
export const CHAT_WEB_FETCH_TOOLS_ENABLED_KEY =
  "unsloth_chat_web_fetch_tools_enabled";

/**
 * Built-in OCR model presets selectable from the Document Extraction settings.
 * "default" means: use the loaded chat VLM when it is vision-capable,
 * otherwise behave as no dedicated OCR model.
 * "none" means: no dedicated OCR model override.
 * "custom" means: a user-supplied HF id or local path (see `customOcrModelId`).
 */
export type OcrModelPresetId = "deepseek-ocr" | "glm-ocr" | "paddleocr-vl";
export type OcrModelSelection =
  | OcrModelPresetId
  | "custom"
  | "default"
  | "none";

/**
 * Transient state for the temporary OCR-model swap performed during scanned-PDF
 * extraction. Lives in the store (not localStorage) so the settings sheet, the
 * composer, and the chat header can all subscribe to a single source of truth.
 */
export type OcrPhase =
  | "idle"
  | "validating"
  | "unloading"
  | "loading_ocr"
  | "extracting"
  | "restoring"
  | "error";

export interface DocExtractSettings {
  /** Global on/off for document-drop extraction. */
  enabled: boolean;
  /** Caption extracted visual payloads using the currently loaded vision model. */
  describeImages: boolean;
  /** Render full-page visual payloads for scanned PDFs without a text layer. */
  useVlmOcr: boolean;
  /** Upper bound on figure/page references listed per document. */
  maxFigures: number;
  /** Upper bound on extracted image bytes sent with a document. */
  maxVisualPayloads: number;
  /** Approx chars/4 token budget injected into the outgoing message. */
  tokenBudget: number;
  /**
   * Selected OCR model. "default" follows the loaded VLM if present;
   * "none" keeps the OCR override empty; a preset id loads that preset;
   * "custom" reads from `customOcrModelId`.
   */
  ocrModel: OcrModelSelection;
  /** HF id or absolute local path used when `ocrModel === "custom"`. */
  customOcrModelId: string;
  /** GGUF variant filename for custom OCR repos that ship GGUF; null otherwise. */
  customOcrGgufVariant: string | null;
  /**
   * Frontend-side cap on parallel `/chat/extract-document` requests.
   * Mirrors the backend `_EXTRACT_SEMAPHORE` so dropping many files at
   * once queues client-side instead of producing 503-busy responses.
   */
  extractConcurrency: number;
}

export const DEFAULT_DOC_EXTRACT: DocExtractSettings = {
  enabled: true,
  describeImages: true,
  useVlmOcr: false,
  maxFigures: 40,
  maxVisualPayloads: DEFAULT_DOCUMENT_VISUAL_PAYLOADS,
  tokenBudget: 8000,
  ocrModel: "default",
  customOcrModelId: "",
  customOcrGgufVariant: null,
  extractConcurrency: DEFAULT_EXTRACT_CONCURRENCY,
};

function clampExtractConcurrency(value: unknown): number {
  const n =
    typeof value === "number" && Number.isFinite(value)
      ? Math.floor(value)
      : DEFAULT_EXTRACT_CONCURRENCY;
  return Math.max(1, Math.min(MAX_EXTRACT_CONCURRENCY, n));
}

const VALID_OCR_SELECTIONS: ReadonlySet<OcrModelSelection> = new Set([
  "default",
  "none",
  "custom",
  "deepseek-ocr",
  "glm-ocr",
  "paddleocr-vl",
]);

function asOcrSelection(value: unknown): OcrModelSelection {
  return typeof value === "string" &&
    VALID_OCR_SELECTIONS.has(value as OcrModelSelection)
    ? (value as OcrModelSelection)
    : DEFAULT_DOC_EXTRACT.ocrModel;
}

// External provider selection is encoded into `params.checkpoint` as
// `external::<providerId>::<modelId>`. PersistedChatSettings deliberately
// Omits `checkpoint` because the local-model side is mirrored by the
// backend's `/api/inference/status.active_model` response. External
// selections have no such backend mirror, so without explicit
// localStorage persistence here the user's external pick is silently
// reset to the default on every page refresh.
const LAST_EXTERNAL_CHECKPOINT_KEY = "unsloth_chat_last_external_checkpoint";

function loadLastExternalCheckpoint(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const value = window.localStorage.getItem(LAST_EXTERNAL_CHECKPOINT_KEY);
    return isExternalModelId(value) ? value : null;
  } catch {
    return null;
  }
}
function saveLastExternalCheckpoint(value: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (value && isExternalModelId(value)) {
      window.localStorage.setItem(LAST_EXTERNAL_CHECKPOINT_KEY, value);
    } else {
      // Clearing on a switch to a local / empty checkpoint means the
      // next refresh won't override the now-active local selection.
      window.localStorage.removeItem(LAST_EXTERNAL_CHECKPOINT_KEY);
    }
  } catch {
    // Storage quota / private-mode failures are non-fatal -- the
    // selection just won't survive the refresh.
  }
}

export type ReasoningStyle = "enable_thinking" | "reasoning_effort";
export type PendingImageEditReference = {
  threadId: string | null;
  openaiImageGenerationCallId: string;
  openaiResponseId?: string;
  openaiReasoningItem?: unknown;
};
export type ReasoningEffort =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "max"
  | "xhigh";

let hasShownInferencePersistenceWarning = false;
let hasShownStoragePersistenceWarning = false;
let hasShownSettingsPersistenceWarning = false;
let customPresetsMutationVersion = 0;
let activePresetMutationVersion = 0;
let activePresetSourceMutationVersion = 0;
let settingsHydrationPromise: Promise<void> | null = null;

function warnSettingsPersistenceFailure(): void {
  if (hasShownSettingsPersistenceWarning) {
    return;
  }
  hasShownSettingsPersistenceWarning = true;
  toast.warning("Chat settings could not be persisted", {
    description: "Your changes apply now, but may reset after refresh.",
  });
}

// Coalesce setting writes into one pendingPatch (deep merge for nested
// keys), flush on a trailing-edge debounce, flush on beforeunload so a
// pending patch survives tab close. Slider drag ticks now produce one
// HTTP write per quiet window instead of one per tick.
type SettingsPatch = Parameters<typeof savePersistedChatSettingsPatch>[0];

const SETTINGS_DEBOUNCE_MS = 400;
let pendingPatch: SettingsPatch = {};
let pendingTimer: ReturnType<typeof setTimeout> | null = null;
let inflightFlush: Promise<void> = Promise.resolve();

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function mergePatch(into: SettingsPatch, more: SettingsPatch): void {
  for (const [key, value] of Object.entries(more)) {
    const intoAny = into as Record<string, unknown>;
    const prev = intoAny[key];
    if (isPlainObject(prev) && isPlainObject(value)) {
      intoAny[key] = { ...prev, ...value };
    } else {
      intoAny[key] = value;
    }
  }
}

async function flushSettingsPatch(keepalive = false): Promise<void> {
  if (Object.keys(pendingPatch).length === 0) return;
  const patch = pendingPatch;
  pendingPatch = {};
  try {
    await savePersistedChatSettingsPatch(patch, { keepalive });
  } catch {
    const retryPatch: SettingsPatch = {};
    mergePatch(retryPatch, patch);
    mergePatch(retryPatch, pendingPatch);
    pendingPatch = retryPatch;
    warnSettingsPersistenceFailure();
  }
}

function saveSettingsPatch(patch: SettingsPatch): void {
  mergePatch(pendingPatch, patch);
  if (pendingTimer !== null) clearTimeout(pendingTimer);
  pendingTimer = setTimeout(() => {
    pendingTimer = null;
    inflightFlush = inflightFlush
      .catch(() => undefined)
      .then(() => flushSettingsPatch());
  }, SETTINGS_DEBOUNCE_MS);
}

// Best-effort flush of any pending patch when the tab closes. keepalive
// lets the PUT outlive the unload; without it the browser cancels the
// fetch and the user's last slider drag is dropped.
if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", () => {
    if (pendingTimer !== null) clearTimeout(pendingTimer);
    if (Object.keys(pendingPatch).length === 0) return;
    inflightFlush = inflightFlush
      .catch(() => undefined)
      .then(() => flushSettingsPatch(true));
  });
}

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function loadBool(key: string, fallback: boolean): boolean {
  const raw = loadOptionalBool(key);
  return raw ?? fallback;
}

export function loadOptionalBool(key: string): boolean | null {
  if (!canUseStorage()) return null;
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return null;
    return raw === "true";
  } catch {
    return null;
  }
}

function warnStoragePersistence(): void {
  if (hasShownStoragePersistenceWarning) return;
  hasShownStoragePersistenceWarning = true;
  toast.warning("Chat settings could not be persisted", {
    description: "Your changes apply now, but may reset after refresh.",
  });
}

function saveBool(key: string, value: boolean): boolean {
  if (!canUseStorage()) return false;
  try {
    localStorage.setItem(key, value ? "true" : "false");
    return true;
  } catch {
    return false;
  }
}

function loadInt(key: string, fallback: number): number {
  if (!canUseStorage()) return fallback;
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    const parsed = Number.parseInt(raw, 10);
    return Number.isNaN(parsed) ? fallback : parsed;
  } catch {
    return fallback;
  }
}

function saveInt(key: string, value: number): boolean {
  if (!canUseStorage()) return false;
  try {
    localStorage.setItem(key, String(value));
    return true;
  } catch {
    return false;
  }
}

function loadString(key: string, fallback: string): string {
  if (!canUseStorage()) return fallback;
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function saveString(key: string, value: string): boolean {
  if (!canUseStorage()) return false;
  try {
    localStorage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

function asFiniteNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asNonNegativeInteger(value: unknown, fallback: number): number {
  return Math.max(0, Math.round(asFiniteNumber(value, fallback)));
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function loadInferenceParams(): InferenceParams {
  if (!canUseStorage()) return DEFAULT_INFERENCE_PARAMS;
  try {
    const raw = localStorage.getItem(INFERENCE_PARAMS_KEY);
    if (!raw) return DEFAULT_INFERENCE_PARAMS;
    const parsed = JSON.parse(raw) as Partial<InferenceParams>;
    return {
      temperature: asFiniteNumber(
        parsed.temperature,
        DEFAULT_INFERENCE_PARAMS.temperature,
      ),
      topP: asFiniteNumber(parsed.topP, DEFAULT_INFERENCE_PARAMS.topP),
      topK: asFiniteNumber(parsed.topK, DEFAULT_INFERENCE_PARAMS.topK),
      minP: asFiniteNumber(parsed.minP, DEFAULT_INFERENCE_PARAMS.minP),
      repetitionPenalty: asFiniteNumber(
        parsed.repetitionPenalty,
        DEFAULT_INFERENCE_PARAMS.repetitionPenalty,
      ),
      presencePenalty: asFiniteNumber(
        parsed.presencePenalty,
        DEFAULT_INFERENCE_PARAMS.presencePenalty,
      ),
      maxSeqLength: asFiniteNumber(
        parsed.maxSeqLength,
        DEFAULT_INFERENCE_PARAMS.maxSeqLength,
      ),
      maxTokens: asFiniteNumber(
        parsed.maxTokens,
        DEFAULT_INFERENCE_PARAMS.maxTokens,
      ),
      systemPrompt: asString(
        parsed.systemPrompt,
        DEFAULT_INFERENCE_PARAMS.systemPrompt,
      ),
      checkpoint: DEFAULT_INFERENCE_PARAMS.checkpoint,
      trustRemoteCode: asBoolean(
        parsed.trustRemoteCode,
        DEFAULT_INFERENCE_PARAMS.trustRemoteCode ?? false,
      ),
    };
  } catch {
    return DEFAULT_INFERENCE_PARAMS;
  }
}

function saveInferenceParams(params: InferenceParams): boolean {
  if (!canUseStorage()) return false;
  try {
    const { checkpoint, ...rest } = params;
    void checkpoint;
    localStorage.setItem(INFERENCE_PARAMS_KEY, JSON.stringify(rest));
    return true;
  } catch {
    return false;
  }
}

function loadPresetSource(): ChatPresetSource {
  const activePreset = loadString(CHAT_ACTIVE_PRESET_KEY, "Default");
  if (canUseStorage()) {
    try {
      const raw = localStorage.getItem(CHAT_ACTIVE_PRESET_SOURCE_KEY);
      if (raw === "modified") {
        return "modified";
      }
    } catch {
      // ignore
    }
  }
  return getPresetSource(activePreset);
}

function loadDocExtract(): DocExtractSettings {
  if (!canUseStorage()) return DEFAULT_DOC_EXTRACT;
  try {
    const raw = localStorage.getItem(DOC_EXTRACT_KEY);
    if (!raw) return DEFAULT_DOC_EXTRACT;
    const parsed = JSON.parse(raw) as Partial<DocExtractSettings>;
    return {
      enabled: asBoolean(parsed.enabled, DEFAULT_DOC_EXTRACT.enabled),
      describeImages: asBoolean(
        parsed.describeImages,
        DEFAULT_DOC_EXTRACT.describeImages,
      ),
      useVlmOcr: asBoolean(parsed.useVlmOcr, DEFAULT_DOC_EXTRACT.useVlmOcr),
      maxFigures: asNonNegativeInteger(
        parsed.maxFigures,
        DEFAULT_DOC_EXTRACT.maxFigures,
      ),
      maxVisualPayloads: asNonNegativeInteger(
        parsed.maxVisualPayloads,
        DEFAULT_DOC_EXTRACT.maxVisualPayloads,
      ),
      tokenBudget: asNonNegativeInteger(
        parsed.tokenBudget,
        DEFAULT_DOC_EXTRACT.tokenBudget,
      ),
      ocrModel: asOcrSelection(parsed.ocrModel),
      customOcrModelId: asString(
        parsed.customOcrModelId,
        DEFAULT_DOC_EXTRACT.customOcrModelId,
      ),
      customOcrGgufVariant:
        typeof parsed.customOcrGgufVariant === "string"
          ? parsed.customOcrGgufVariant
          : DEFAULT_DOC_EXTRACT.customOcrGgufVariant,
      extractConcurrency: clampExtractConcurrency(parsed.extractConcurrency),
    };
  } catch {
    return DEFAULT_DOC_EXTRACT;
  }
}

function saveDocExtract(value: DocExtractSettings): boolean {
  if (!canUseStorage()) return false;
  try {
    localStorage.setItem(DOC_EXTRACT_KEY, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

type ChatRuntimeStore = {
  settingsHydrated: boolean;
  params: InferenceParams;
  customPresets: Preset[];
  activePreset: string;
  activePresetSource: ChatPresetSource;
  models: ChatModelSummary[];
  loras: ChatLoraSummary[];
  runningByThreadId: Record<string, boolean>;
  cancelByThreadId: Record<string, () => void>;
  autoTitle: boolean;
  hfToken: string;
  modelsError: string | null;
  activeGgufVariant: string | null;
  ggufContextLength: number | null;
  ggufMaxContextLength: number | null;
  ggufNativeContextLength: number | null;
  modelRequiresTrustRemoteCode: boolean;
  supportsReasoning: boolean;
  reasoningAlwaysOn: boolean;
  reasoningEnabled: boolean;
  /**
   * The model id the OpenRouter router actually picked for the most recent
   * stream when the active checkpoint is the openrouter/free meta-model.
   * Updated each time a chunk arrives carrying a non-empty `model` field
   * that differs from the requested id. Cleared when a non-OpenRouter
   * model is selected. Used purely for UI display — appended after
   * `openrouter/free:` in the active model chip.
   */
  lastOpenRouterChosenModel: string | null;
  reasoningStyle: ReasoningStyle;
  reasoningEffort: ReasoningEffort;
  supportsReasoningOff: boolean;
  reasoningEffortLevels: readonly ReasoningEffort[];
  supportsPreserveThinking: boolean;
  preserveThinking: boolean;
  supportsTools: boolean;
  /**
   * Whether the active external provider exposes a server-side
   * web_search tool (OpenAI's /v1/responses today). Distinct from
   * `supportsTools` — that flag governs the local tool runtime (Code,
   * python sandbox, our DuckDuckGo web_search). This one only enables
   * the chat composer's Search pill for external models. Local models
   * keep `supportsTools` only.
   */
  supportsBuiltinWebSearch: boolean;
  /**
   * Whether the active external provider exposes a server-side
   * code-execution tool (Anthropic's `code_execution_20250825` on the
   * Claude 4.x family). Distinct from `supportsTools` for the same
   * reason as `supportsBuiltinWebSearch`: external providers don't
   * give us a local tool runtime, but Anthropic dispatches code
   * execution server-side. Read by both composers' Code pill gate.
   */
  supportsBuiltinCodeExecution: boolean;
  /**
   * Whether the active external provider exposes a server-side
   * image-generation tool (OpenAI's Responses-API `image_generation`
   * today). Gates the chat composer's Images pill. Local models never
   * receive the tool because their runtime cannot dispatch it.
   */
  supportsBuiltinImageGeneration: boolean;
  /**
   * Whether the active external provider exposes a server-side
   * web_fetch tool (Anthropic's `web_fetch_20250910` /
   * `web_fetch_20260209`). Gates the composer's Fetch pill,
   * independent of Search.
   */
  supportsBuiltinWebFetch: boolean;
  toolsEnabled: boolean;
  codeToolsEnabled: boolean;
  imageToolsEnabled: boolean;
  /**
   * Fetch pill state, independent of `toolsEnabled` (Search). Only
   * consulted when `providerSupportsBuiltinWebFetch` is true.
   */
  webFetchToolsEnabled: boolean;
  toolStatus: string | null;
  generatingStatus: string | null;
  autoHealToolCalls: boolean;
  maxToolCallsPerMessage: number;
  toolCallTimeout: number;
  kvCacheDtype: string | null;
  loadedKvCacheDtype: string | null;
  speculativeType: string | null;
  loadedSpeculativeType: string | null;
  /** User --spec-draft-n-max override (null = platform default). */
  specDraftNMax: number | null;
  loadedSpecDraftNMax: number | null;
  loadedIsMultimodal: boolean;
  customContextLength: number | null;
  defaultChatTemplate: string | null;
  chatTemplateOverride: string | null;
  loadedChatTemplateOverride: string | null;
  activeThreadId: string | null;
  settingsPanelOpen: boolean;
  pendingAudioBase64: string | null;
  pendingAudioName: string | null;
  pendingImageEditReference: PendingImageEditReference | null;
  contextUsage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    cachedTokens: number;
    // Anthropic-only; optional so pre-cache-stats persisted entries load.
    cacheWriteTokens?: number;
  } | null;
  modelLoading: boolean;
  activeNativePathToken: string | null;
  docExtract: DocExtractSettings;
  ocrPhase: OcrPhase;
  setDocExtract: (value: Partial<DocExtractSettings>) => void;
  setOcrPhase: (phase: OcrPhase) => void;
  hydratePersistedSettings: () => Promise<void>;
  setModelLoading: (loading: boolean) => void;
  setModelRequiresTrustRemoteCode: (required: boolean) => void;
  setParams: (params: InferenceParams) => void;
  setCustomPresets: (presets: Preset[]) => void;
  setActivePreset: (name: string) => void;
  setActivePresetSource: (source: ChatPresetSource) => void;
  setModels: (models: ChatModelSummary[]) => void;
  setLoras: (loras: ChatLoraSummary[]) => void;
  setThreadRunning: (threadId: string, running: boolean) => void;
  registerThreadCancel: (threadId: string, cancel: () => void) => void;
  clearThreadCancel: (threadId: string) => void;
  setAutoTitle: (enabled: boolean) => void;
  setHfToken: (token: string) => void;
  setModelsError: (error: string | null) => void;
  setCheckpoint: (modelId: string, ggufVariant?: string | null) => void;
  setActiveThreadId: (threadId: string | null) => void;
  setSettingsPanelOpen: (open: boolean) => void;
  clearCheckpoint: () => void;
  setReasoningEnabled: (
    enabled: boolean,
    options?: { persist?: boolean },
  ) => void;
  setLastOpenRouterChosenModel: (chosen: string | null) => void;
  setReasoningStyle: (style: ReasoningStyle) => void;
  setReasoningEffort: (effort: ReasoningEffort) => void;
  setPreserveThinking: (value: boolean) => void;
  setToolsEnabled: (enabled: boolean, options?: { persist?: boolean }) => void;
  setCodeToolsEnabled: (enabled: boolean) => void;
  setImageToolsEnabled: (enabled: boolean) => void;
  setWebFetchToolsEnabled: (enabled: boolean) => void;
  setToolStatus: (status: string | null) => void;
  setGeneratingStatus: (status: string | null) => void;
  setAutoHealToolCalls: (enabled: boolean) => void;
  setMaxToolCallsPerMessage: (value: number) => void;
  setToolCallTimeout: (value: number) => void;
  setKvCacheDtype: (dtype: string | null) => void;
  setSpeculativeType: (type: string | null) => void;
  setSpecDraftNMax: (value: number | null) => void;
  setCustomContextLength: (v: number | null) => void;
  setChatTemplateOverride: (template: string | null) => void;
  setPendingAudio: (base64: string, name: string) => void;
  clearPendingAudio: () => void;
  setPendingImageEditReference: (
    reference: PendingImageEditReference | null,
  ) => void;
  clearPendingImageEditReference: () => void;
  setContextUsage: (usage: ChatRuntimeStore["contextUsage"]) => void;
};

type PersistedChatSettings = Awaited<
  ReturnType<typeof loadChatSettingsWithLegacyImport>
>;
type PersistedInferenceParams = NonNullable<
  PersistedChatSettings["inferenceParams"]
>;
type PersistedInferenceParamKey = keyof PersistedInferenceParams;
type ScalarSettingKey =
  | "autoTitle"
  | "reasoningEffort"
  | "preserveThinking"
  | "autoHealToolCalls"
  | "maxToolCallsPerMessage"
  | "toolCallTimeout";

type PresetHydrationVersions = {
  customPresets: number;
  activePreset: number;
  activePresetSource: number;
};

type SettingsHydrationVersions = {
  inferenceParams: Record<PersistedInferenceParamKey, number>;
  scalarSettings: Record<ScalarSettingKey, number>;
  presets: PresetHydrationVersions;
};

const PERSISTED_INFERENCE_PARAM_KEYS = [
  "temperature",
  "topP",
  "topK",
  "minP",
  "repetitionPenalty",
  "presencePenalty",
  "maxSeqLength",
  "maxTokens",
  "systemPrompt",
  "trustRemoteCode",
  "fastMode",
] as const satisfies readonly PersistedInferenceParamKey[];

const SCALAR_SETTING_KEYS = [
  "autoTitle",
  "reasoningEffort",
  "preserveThinking",
  "autoHealToolCalls",
  "maxToolCallsPerMessage",
  "toolCallTimeout",
] as const satisfies readonly ScalarSettingKey[];

const inferenceParamMutationVersions = Object.fromEntries(
  PERSISTED_INFERENCE_PARAM_KEYS.map((key) => [key, 0]),
) as Record<PersistedInferenceParamKey, number>;
const scalarSettingMutationVersions = Object.fromEntries(
  SCALAR_SETTING_KEYS.map((key) => [key, 0]),
) as Record<ScalarSettingKey, number>;

function hasKeys(value: object): boolean {
  return Object.keys(value).length > 0;
}

function getSettingsHydrationVersions(): SettingsHydrationVersions {
  return {
    inferenceParams: { ...inferenceParamMutationVersions },
    scalarSettings: { ...scalarSettingMutationVersions },
    presets: {
      customPresets: customPresetsMutationVersion,
      activePreset: activePresetMutationVersion,
      activePresetSource: activePresetSourceMutationVersion,
    },
  };
}

function setInferenceParam(
  params: InferenceParams,
  key: PersistedInferenceParamKey,
  value: PersistedInferenceParams[PersistedInferenceParamKey],
): void {
  (params as Record<PersistedInferenceParamKey, unknown>)[key] = value;
}

function getChangedInferenceParams(
  nextParams: InferenceParams,
  currentParams: InferenceParams,
): PersistedInferenceParams {
  const changedParams: PersistedInferenceParams = {};
  for (const key of PERSISTED_INFERENCE_PARAM_KEYS) {
    const nextValue = nextParams[key];
    if (Object.is(nextValue, currentParams[key])) {
      continue;
    }
    inferenceParamMutationVersions[key] += 1;
    if (nextValue !== undefined) {
      setInferenceParam(changedParams as InferenceParams, key, nextValue);
    }
  }
  return changedParams;
}

function getHydratedCustomPresets(
  settings: PersistedChatSettings,
  state: ChatRuntimeStore,
): Preset[] {
  return (
    settings.customPresets?.map((preset) => ({
      name: preset.name,
      params: {
        ...DEFAULT_INFERENCE_PARAMS,
        ...preset.params,
      },
    })) ?? state.customPresets
  );
}

function getHydratedPresetState(
  settings: PersistedChatSettings,
  state: ChatRuntimeStore,
  versions: PresetHydrationVersions,
): Partial<
  Pick<
    ChatRuntimeStore,
    "customPresets" | "activePreset" | "activePresetSource"
  >
> {
  const nextState: Partial<
    Pick<
      ChatRuntimeStore,
      "customPresets" | "activePreset" | "activePresetSource"
    >
  > = {};
  if (customPresetsMutationVersion === versions.customPresets) {
    nextState.customPresets = getHydratedCustomPresets(settings, state);
  }
  if (activePresetMutationVersion === versions.activePreset) {
    nextState.activePreset = settings.activePreset ?? state.activePreset;
  }
  if (activePresetSourceMutationVersion === versions.activePresetSource) {
    const activePreset = nextState.activePreset ?? state.activePreset;
    nextState.activePresetSource =
      settings.activePresetSource ?? getPresetSource(activePreset);
  }
  return nextState;
}

function getHydratedSettingsState(
  settings: PersistedChatSettings,
  state: ChatRuntimeStore,
  versions: SettingsHydrationVersions,
): Partial<ChatRuntimeStore> {
  const nextState: Partial<ChatRuntimeStore> = {};
  const params = { ...state.params };
  for (const key of PERSISTED_INFERENCE_PARAM_KEYS) {
    const value = settings.inferenceParams?.[key];
    if (
      value !== undefined &&
      inferenceParamMutationVersions[key] === versions.inferenceParams[key]
    ) {
      setInferenceParam(params, key, value);
    }
  }
  nextState.params = params;
  for (const key of SCALAR_SETTING_KEYS) {
    const value = settings[key];
    if (
      value !== undefined &&
      scalarSettingMutationVersions[key] === versions.scalarSettings[key]
    ) {
      (nextState as Record<ScalarSettingKey, unknown>)[key] = value;
    }
  }
  return nextState;
}

function setScalarSettingVersion<K extends ScalarSettingKey>(
  key: K,
  value: ChatRuntimeStore[K],
  currentValue: ChatRuntimeStore[K],
): void {
  if (Object.is(value, currentValue)) {
    return;
  }
  scalarSettingMutationVersions[key] += 1;
  saveSettingsPatch({ [key]: value });
}

export const useChatRuntimeStore = create<ChatRuntimeStore>((set, get) => ({
  settingsHydrated: false,
  // Hydrate the last external checkpoint into params.checkpoint so the
  // external picker selection survives a page refresh. Local model
  // checkpoints are re-derived from the backend in useChatModelRuntime
  // and intentionally NOT persisted here.
  params: (() => {
    const persistedExternal = loadLastExternalCheckpoint();
    return persistedExternal
      ? { ...DEFAULT_INFERENCE_PARAMS, checkpoint: persistedExternal }
      : DEFAULT_INFERENCE_PARAMS;
  })(),
  customPresets: [],
  activePreset: "Default",
  activePresetSource: getPresetSource("Default"),
  models: [],
  loras: [],
  runningByThreadId: {},
  cancelByThreadId: {},
  autoTitle: false,
  hfToken: loadString(HF_TOKEN_KEY, ""),
  modelsError: null,
  activeGgufVariant: null,
  ggufContextLength: null,
  ggufMaxContextLength: null,
  ggufNativeContextLength: null,
  modelRequiresTrustRemoteCode: false,
  supportsReasoning: false,
  reasoningAlwaysOn: false,
  reasoningEnabled: loadBool(CHAT_REASONING_ENABLED_KEY, true),
  reasoningStyle: "enable_thinking",
  reasoningEffort: "medium",
  supportsReasoningOff: false,
  reasoningEffortLevels: ["low", "medium", "high"],
  lastOpenRouterChosenModel: null,
  supportsPreserveThinking: false,
  preserveThinking: false,
  supportsTools: false,
  supportsBuiltinWebSearch: false,
  supportsBuiltinCodeExecution: false,
  supportsBuiltinImageGeneration: false,
  supportsBuiltinWebFetch: false,
  toolsEnabled: loadBool(CHAT_TOOLS_ENABLED_KEY, false),
  codeToolsEnabled: loadBool(CHAT_CODE_TOOLS_ENABLED_KEY, false),
  imageToolsEnabled: loadBool(CHAT_IMAGE_TOOLS_ENABLED_KEY, false),
  webFetchToolsEnabled: loadBool(CHAT_WEB_FETCH_TOOLS_ENABLED_KEY, false),
  toolStatus: null,
  generatingStatus: null,
  autoHealToolCalls: true,
  maxToolCallsPerMessage: 25,
  toolCallTimeout: 5,
  kvCacheDtype: null,
  loadedKvCacheDtype: null,
  speculativeType: "auto",
  loadedSpeculativeType: null,
  specDraftNMax: null,
  loadedSpecDraftNMax: null,
  loadedIsMultimodal: false,
  customContextLength: null,
  defaultChatTemplate: null,
  chatTemplateOverride: null,
  loadedChatTemplateOverride: null,
  activeThreadId: null,
  settingsPanelOpen: false,
  pendingAudioBase64: null,
  pendingAudioName: null,
  pendingImageEditReference: null,
  contextUsage: null,
  modelLoading: false,
  activeNativePathToken: null,
  docExtract: loadDocExtract(),
  ocrPhase: "idle",
  setDocExtract: (value) =>
    set((state) => {
      const merged = { ...state.docExtract, ...value };
      const next: DocExtractSettings = {
        ...merged,
        extractConcurrency: clampExtractConcurrency(merged.extractConcurrency),
      };
      if (!saveDocExtract(next)) {
        warnStoragePersistence();
      }
      return { docExtract: next };
    }),
  setOcrPhase: (ocrPhase) => set({ ocrPhase }),
  hydratePersistedSettings: async () => {
    if (get().settingsHydrated) {
      return;
    }
    if (settingsHydrationPromise) {
      return settingsHydrationPromise;
    }
    settingsHydrationPromise = (async () => {
      const hydrationVersions = getSettingsHydrationVersions();
      try {
        const settings = await loadChatSettingsWithLegacyImport();
        set((state) => {
          if (state.settingsHydrated) {
            return state;
          }
          const nextState: Partial<ChatRuntimeStore> = {
            settingsHydrated: true,
            ...getHydratedPresetState(
              settings,
              state,
              hydrationVersions.presets,
            ),
            ...getHydratedSettingsState(settings, state, hydrationVersions),
          };
          return nextState;
        });
      } catch {
        // Hydrate failed: treat as hydrated-with-defaults so future
        // setParams calls reach saveSettingsPatch (which surfaces its
        // own toast on real network failure).
        warnSettingsPersistenceFailure();
        set({ settingsHydrated: true });
      } finally {
        settingsHydrationPromise = null;
      }
    })();
    return settingsHydrationPromise;
  },
  setModelLoading: (loading) => set({ modelLoading: loading }),
  setModelRequiresTrustRemoteCode: (modelRequiresTrustRemoteCode) =>
    set({ modelRequiresTrustRemoteCode }),
  setParams: (params) =>
    set((state) => {
      // Bump version unconditionally so a late hydration response
      // won't clobber a pre-hydrate user edit; only the HTTP write
      // is gated on settingsHydrated.
      const changedParams = getChangedInferenceParams(params, state.params);
      if (state.settingsHydrated && hasKeys(changedParams)) {
        saveSettingsPatch({ inferenceParams: changedParams });
      }
      // Mirror setCheckpoint: the local model load path can mutate
      // params.checkpoint via setParams() before setCheckpoint runs,
      // leaving stale per-turn counters under the new checkpoint.
      const checkpointChanged = state.params.checkpoint !== params.checkpoint;
      return {
        params,
        ...(checkpointChanged ? { contextUsage: null } : {}),
      };
    }),
  setCustomPresets: (customPresets) =>
    set(() => {
      customPresetsMutationVersion += 1;
      saveSettingsPatch({ customPresets });
      return { customPresets };
    }),
  setActivePreset: (activePreset) =>
    set(() => {
      activePresetMutationVersion += 1;
      saveSettingsPatch({ activePreset });
      return { activePreset };
    }),
  setActivePresetSource: (activePresetSource) =>
    set(() => {
      activePresetSourceMutationVersion += 1;
      saveSettingsPatch({ activePresetSource });
      return { activePresetSource };
    }),
  setModels: (models) => set({ models }),
  setLoras: (loras) => set({ loras }),
  setThreadRunning: (threadId, running) =>
    set((state) => {
      const next = { ...state.runningByThreadId };
      if (running) {
        next[threadId] = true;
      } else {
        delete next[threadId];
      }
      return { runningByThreadId: next };
    }),
  registerThreadCancel: (threadId, cancel) =>
    set((state) => {
      const next = { ...state.cancelByThreadId };
      next[threadId] = cancel;
      return { cancelByThreadId: next };
    }),
  clearThreadCancel: (threadId) =>
    set((state) => {
      if (!(threadId in state.cancelByThreadId)) return state;
      const next = { ...state.cancelByThreadId };
      delete next[threadId];
      return { cancelByThreadId: next };
    }),
  setAutoTitle: (autoTitle) =>
    set((state) => {
      setScalarSettingVersion("autoTitle", autoTitle, state.autoTitle);
      return { autoTitle };
    }),
  setHfToken: (hfToken) =>
    set(() => {
      if (!saveString(HF_TOKEN_KEY, hfToken)) {
        warnStoragePersistence();
      }
      return { hfToken };
    }),
  setModelsError: (modelsError) => set({ modelsError }),
  setCheckpoint: (modelId, ggufVariant) =>
    set((state) => {
      invalidateDocumentSupportCache();
      // Persist external selections so they survive a page refresh.
      // Local model ids are NOT persisted here - they get re-derived
      // from the backend's `/api/inference/status.active_model` on
      // mount, and a stale persisted local id would race against the
      // freshly-loaded model. See LAST_EXTERNAL_CHECKPOINT_KEY notes.
      saveLastExternalCheckpoint(isExternalModelId(modelId) ? modelId : null);
      // Clear stale per-turn usage when the model changes; the relaxed
      // external-provider render gate would otherwise show old counters
      // until the next completion overwrites them.
      const checkpointChanged = state.params.checkpoint !== modelId;
      // Clamp maxTokens to the new model's cap on switch into an
      // external model so a value carried over from a prior local
      // session does not render above the slider's max.
      let nextMaxTokens = state.params.maxTokens;
      if (checkpointChanged && isExternalModelId(modelId)) {
        const parsed = parseExternalModelId(modelId);
        const provider = parsed
          ? useExternalProvidersStore
              .getState()
              .providers.find((p) => p.id === parsed.providerId)
          : null;
        const cap = getExternalMaxOutputTokens(
          provider?.providerType,
          parsed?.modelId,
        );
        if (nextMaxTokens > cap) {
          nextMaxTokens = cap;
        }
      }
      return {
        params: {
          ...state.params,
          checkpoint: modelId,
          maxTokens: nextMaxTokens,
        },
        activeGgufVariant: ggufVariant ?? null,
        ...(checkpointChanged ? { contextUsage: null } : {}),
      };
    }),
  setActiveThreadId: (activeThreadId) =>
    set({ activeThreadId, contextUsage: null }),
  setSettingsPanelOpen: (settingsPanelOpen) => set({ settingsPanelOpen }),
  clearCheckpoint: () => {
    invalidateDocumentSupportCache();
    // Mirror setCheckpoint's persistence behavior: dropping the
    // checkpoint must also clear any stored external selection so
    // the next refresh doesn't snap back to a model the user
    // intentionally cleared.
    saveLastExternalCheckpoint(null);
    return set((state) => ({
      params: {
        ...state.params,
        checkpoint: "",
      },
      activeGgufVariant: null,
      activeNativePathToken: null,
      ggufContextLength: null,
      ggufMaxContextLength: null,
      ggufNativeContextLength: null,
      modelRequiresTrustRemoteCode: false,
      contextUsage: null,
      supportsReasoning: false,
      reasoningAlwaysOn: false,
      reasoningEnabled: true,
      reasoningStyle: "enable_thinking",
      supportsReasoningOff: false,
      reasoningEffortLevels: ["low", "medium", "high"],
      supportsPreserveThinking: false,
      supportsTools: false,
      supportsBuiltinWebSearch: false,
      supportsBuiltinCodeExecution: false,
      supportsBuiltinImageGeneration: false,
      supportsBuiltinWebFetch: false,
      toolsEnabled: false,
      codeToolsEnabled: false,
      imageToolsEnabled: false,
      webFetchToolsEnabled: false,
      toolStatus: null,
      kvCacheDtype: null,
      loadedKvCacheDtype: null,
      speculativeType: "auto",
      loadedSpeculativeType: null,
      specDraftNMax: null,
      loadedSpecDraftNMax: null,
      loadedIsMultimodal: false,
      customContextLength: null,
      defaultChatTemplate: null,
      chatTemplateOverride: null,
      loadedChatTemplateOverride: null,
      pendingImageEditReference: null,
    }));
  },
  setReasoningEnabled: (reasoningEnabled, options) =>
    set(() => {
      if (options?.persist !== false) {
        saveBool(CHAT_REASONING_ENABLED_KEY, reasoningEnabled);
      }
      return { reasoningEnabled };
    }),
  setLastOpenRouterChosenModel: (lastOpenRouterChosenModel) =>
    set({ lastOpenRouterChosenModel }),
  setReasoningStyle: (reasoningStyle) => set({ reasoningStyle }),
  setReasoningEffort: (reasoningEffort) =>
    set((state) => {
      setScalarSettingVersion(
        "reasoningEffort",
        reasoningEffort,
        state.reasoningEffort,
      );
      return { reasoningEffort };
    }),
  setPreserveThinking: (preserveThinking) =>
    set((state) => {
      setScalarSettingVersion(
        "preserveThinking",
        preserveThinking,
        state.preserveThinking,
      );
      return { preserveThinking };
    }),
  setToolsEnabled: (toolsEnabled, options) =>
    set(() => {
      if (options?.persist !== false) {
        saveBool(CHAT_TOOLS_ENABLED_KEY, toolsEnabled);
      }
      return { toolsEnabled };
    }),
  setCodeToolsEnabled: (codeToolsEnabled) =>
    set(() => {
      saveBool(CHAT_CODE_TOOLS_ENABLED_KEY, codeToolsEnabled);
      return { codeToolsEnabled };
    }),
  setImageToolsEnabled: (imageToolsEnabled) =>
    set(() => {
      saveBool(CHAT_IMAGE_TOOLS_ENABLED_KEY, imageToolsEnabled);
      return { imageToolsEnabled };
    }),
  setWebFetchToolsEnabled: (webFetchToolsEnabled) =>
    set(() => {
      saveBool(CHAT_WEB_FETCH_TOOLS_ENABLED_KEY, webFetchToolsEnabled);
      return { webFetchToolsEnabled };
    }),
  setToolStatus: (toolStatus) => set({ toolStatus }),
  setGeneratingStatus: (generatingStatus) => set({ generatingStatus }),
  setAutoHealToolCalls: (autoHealToolCalls) =>
    set((state) => {
      setScalarSettingVersion(
        "autoHealToolCalls",
        autoHealToolCalls,
        state.autoHealToolCalls,
      );
      return { autoHealToolCalls };
    }),
  setMaxToolCallsPerMessage: (maxToolCallsPerMessage) =>
    set((state) => {
      setScalarSettingVersion(
        "maxToolCallsPerMessage",
        maxToolCallsPerMessage,
        state.maxToolCallsPerMessage,
      );
      return { maxToolCallsPerMessage };
    }),
  setToolCallTimeout: (toolCallTimeout) =>
    set((state) => {
      setScalarSettingVersion(
        "toolCallTimeout",
        toolCallTimeout,
        state.toolCallTimeout,
      );
      return { toolCallTimeout };
    }),
  setKvCacheDtype: (kvCacheDtype) => set({ kvCacheDtype }),
  setSpeculativeType: (speculativeType) => set({ speculativeType }),
  setSpecDraftNMax: (specDraftNMax) => set({ specDraftNMax }),
  setCustomContextLength: (customContextLength) => set({ customContextLength }),
  setChatTemplateOverride: (chatTemplateOverride) =>
    set({ chatTemplateOverride }),
  setPendingAudio: (base64, name) =>
    set({ pendingAudioBase64: base64, pendingAudioName: name }),
  clearPendingAudio: () =>
    set({ pendingAudioBase64: null, pendingAudioName: null }),
  setPendingImageEditReference: (pendingImageEditReference) =>
    set({ pendingImageEditReference }),
  clearPendingImageEditReference: () =>
    set({ pendingImageEditReference: null }),
  setContextUsage: (contextUsage) => set({ contextUsage }),
}));
