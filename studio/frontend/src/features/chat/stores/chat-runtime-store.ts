// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";
import { create } from "zustand";
import { invalidateDocumentSupportCache } from "../api/chat-api";
import {
  type ChatLoraSummary,
  type ChatModelSummary,
  DEFAULT_INFERENCE_PARAMS,
  type InferenceParams,
} from "../types/runtime";
import {
  getPresetSource,
  type ChatPresetSource,
} from "../presets/preset-policy";

const AUTO_TITLE_KEY = "unsloth_chat_auto_title";
const AUTO_HEAL_TOOL_CALLS_KEY = "unsloth_auto_heal_tool_calls";
const MAX_TOOL_CALLS_KEY = "unsloth_max_tool_calls_per_message";
const TOOL_CALL_TIMEOUT_KEY = "unsloth_tool_call_timeout";
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

export type ReasoningStyle = "enable_thinking" | "reasoning_effort";
export type ReasoningEffort = "low" | "medium" | "high";

function loadReasoningEffort(fallback: ReasoningEffort): ReasoningEffort {
  if (!canUseStorage()) return fallback;
  try {
    const raw = localStorage.getItem(REASONING_EFFORT_KEY);
    if (raw === "low" || raw === "medium" || raw === "high") return raw;
    return fallback;
  } catch {
    return fallback;
  }
}
let hasShownInferencePersistenceWarning = false;
let hasShownStoragePersistenceWarning = false;

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function loadBool(key: string, fallback: boolean): boolean {
  if (!canUseStorage()) return fallback;
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    return raw === "true";
  } catch {
    return fallback;
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
  params: InferenceParams;
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
  reasoningStyle: ReasoningStyle;
  reasoningEffort: ReasoningEffort;
  supportsPreserveThinking: boolean;
  preserveThinking: boolean;
  supportsTools: boolean;
  toolsEnabled: boolean;
  codeToolsEnabled: boolean;
  toolStatus: string | null;
  generatingStatus: string | null;
  autoHealToolCalls: boolean;
  maxToolCallsPerMessage: number;
  toolCallTimeout: number;
  kvCacheDtype: string | null;
  loadedKvCacheDtype: string | null;
  speculativeType: string | null;
  loadedSpeculativeType: string | null;
  loadedIsMultimodal: boolean;
  customContextLength: number | null;
  defaultChatTemplate: string | null;
  chatTemplateOverride: string | null;
  loadedChatTemplateOverride: string | null;
  activeThreadId: string | null;
  settingsPanelOpen: boolean;
  pendingAudioBase64: string | null;
  pendingAudioName: string | null;
  contextUsage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    cachedTokens: number;
  } | null;
  modelLoading: boolean;
  activeNativePathToken: string | null;
  docExtract: DocExtractSettings;
  ocrPhase: OcrPhase;
  setDocExtract: (value: Partial<DocExtractSettings>) => void;
  setOcrPhase: (phase: OcrPhase) => void;
  setModelLoading: (loading: boolean) => void;
  setModelRequiresTrustRemoteCode: (required: boolean) => void;
  setParams: (params: InferenceParams) => void;
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
  setReasoningEnabled: (enabled: boolean) => void;
  setReasoningStyle: (style: ReasoningStyle) => void;
  setReasoningEffort: (effort: ReasoningEffort) => void;
  setPreserveThinking: (value: boolean) => void;
  setToolsEnabled: (enabled: boolean) => void;
  setCodeToolsEnabled: (enabled: boolean) => void;
  setToolStatus: (status: string | null) => void;
  setGeneratingStatus: (status: string | null) => void;
  setAutoHealToolCalls: (enabled: boolean) => void;
  setMaxToolCallsPerMessage: (value: number) => void;
  setToolCallTimeout: (value: number) => void;
  setKvCacheDtype: (dtype: string | null) => void;
  setSpeculativeType: (type: string | null) => void;
  setCustomContextLength: (v: number | null) => void;
  setChatTemplateOverride: (template: string | null) => void;
  setPendingAudio: (base64: string, name: string) => void;
  clearPendingAudio: () => void;
  setContextUsage: (usage: ChatRuntimeStore["contextUsage"]) => void;
};

export const useChatRuntimeStore = create<ChatRuntimeStore>((set) => ({
  params: loadInferenceParams(),
  activePresetSource: loadPresetSource(),
  models: [],
  loras: [],
  runningByThreadId: {},
  cancelByThreadId: {},
  autoTitle: loadBool(AUTO_TITLE_KEY, false),
  hfToken: loadString(HF_TOKEN_KEY, ""),
  modelsError: null,
  activeGgufVariant: null,
  ggufContextLength: null,
  ggufMaxContextLength: null,
  ggufNativeContextLength: null,
  modelRequiresTrustRemoteCode: false,
  supportsReasoning: false,
  reasoningAlwaysOn: false,
  reasoningEnabled: true,
  reasoningStyle: "enable_thinking",
  reasoningEffort: loadReasoningEffort("medium"),
  supportsPreserveThinking: false,
  preserveThinking: loadBool(PRESERVE_THINKING_KEY, false),
  supportsTools: false,
  toolsEnabled: false,
  codeToolsEnabled: false,
  toolStatus: null,
  generatingStatus: null,
  autoHealToolCalls: loadBool(AUTO_HEAL_TOOL_CALLS_KEY, true),
  maxToolCallsPerMessage: loadInt(MAX_TOOL_CALLS_KEY, 25),
  toolCallTimeout: loadInt(TOOL_CALL_TIMEOUT_KEY, 5),
  kvCacheDtype: null,
  loadedKvCacheDtype: null,
  speculativeType: "default",
  loadedSpeculativeType: null,
  loadedIsMultimodal: false,
  customContextLength: null,
  defaultChatTemplate: null,
  chatTemplateOverride: null,
  loadedChatTemplateOverride: null,
  activeThreadId: null,
  settingsPanelOpen: false,
  pendingAudioBase64: null,
  pendingAudioName: null,
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
  setModelLoading: (loading) => set({ modelLoading: loading }),
  setModelRequiresTrustRemoteCode: (modelRequiresTrustRemoteCode) =>
    set({ modelRequiresTrustRemoteCode }),
  setParams: (params) =>
    set(() => {
      const persisted = saveInferenceParams(params);
      if (!persisted && !hasShownInferencePersistenceWarning) {
        hasShownInferencePersistenceWarning = true;
        toast.warning("Chat settings could not be persisted", {
          description: "Your changes apply now, but may reset after refresh.",
        });
      }
      return { params };
    }),
  setActivePresetSource: (activePresetSource) =>
    set(() => {
      saveString(CHAT_ACTIVE_PRESET_SOURCE_KEY, activePresetSource);
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
    set(() => {
      if (!saveBool(AUTO_TITLE_KEY, autoTitle)) {
        warnStoragePersistence();
      }
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
      return {
        params: {
          ...state.params,
          checkpoint: modelId,
        },
        activeGgufVariant: ggufVariant ?? null,
      };
    }),
  setActiveThreadId: (activeThreadId) =>
    set({ activeThreadId, contextUsage: null }),
  setSettingsPanelOpen: (settingsPanelOpen) => set({ settingsPanelOpen }),
  clearCheckpoint: () =>
    set((state) => {
      invalidateDocumentSupportCache();
      return {
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
        supportsPreserveThinking: false,
        supportsTools: false,
        toolsEnabled: false,
        codeToolsEnabled: false,
        toolStatus: null,
        kvCacheDtype: null,
        loadedKvCacheDtype: null,
        speculativeType: "default",
        loadedSpeculativeType: null,
        loadedIsMultimodal: false,
        customContextLength: null,
        defaultChatTemplate: null,
        chatTemplateOverride: null,
        loadedChatTemplateOverride: null,
      };
    }),
  setReasoningEnabled: (reasoningEnabled) => set({ reasoningEnabled }),
  setReasoningStyle: (reasoningStyle) => set({ reasoningStyle }),
  setReasoningEffort: (reasoningEffort) =>
    set(() => {
      if (!saveString(REASONING_EFFORT_KEY, reasoningEffort)) {
        warnStoragePersistence();
      }
      return { reasoningEffort };
    }),
  setPreserveThinking: (preserveThinking) =>
    set(() => {
      if (!saveBool(PRESERVE_THINKING_KEY, preserveThinking)) {
        warnStoragePersistence();
      }
      return { preserveThinking };
    }),
  setToolsEnabled: (toolsEnabled) => set({ toolsEnabled }),
  setCodeToolsEnabled: (codeToolsEnabled) => set({ codeToolsEnabled }),
  setToolStatus: (toolStatus) => set({ toolStatus }),
  setGeneratingStatus: (generatingStatus) => set({ generatingStatus }),
  setAutoHealToolCalls: (autoHealToolCalls) =>
    set(() => {
      if (!saveBool(AUTO_HEAL_TOOL_CALLS_KEY, autoHealToolCalls)) {
        warnStoragePersistence();
      }
      return { autoHealToolCalls };
    }),
  setMaxToolCallsPerMessage: (maxToolCallsPerMessage) =>
    set(() => {
      if (!saveInt(MAX_TOOL_CALLS_KEY, maxToolCallsPerMessage)) {
        warnStoragePersistence();
      }
      return { maxToolCallsPerMessage };
    }),
  setToolCallTimeout: (toolCallTimeout) =>
    set(() => {
      if (!saveInt(TOOL_CALL_TIMEOUT_KEY, toolCallTimeout)) {
        warnStoragePersistence();
      }
      return { toolCallTimeout };
    }),
  setKvCacheDtype: (kvCacheDtype) => set({ kvCacheDtype }),
  setSpeculativeType: (speculativeType) => set({ speculativeType }),
  setCustomContextLength: (customContextLength) => set({ customContextLength }),
  setChatTemplateOverride: (chatTemplateOverride) =>
    set({ chatTemplateOverride }),
  setPendingAudio: (base64, name) =>
    set({ pendingAudioBase64: base64, pendingAudioName: name }),
  clearPendingAudio: () =>
    set({ pendingAudioBase64: null, pendingAudioName: null }),
  setContextUsage: (contextUsage) => set({ contextUsage }),
}));
