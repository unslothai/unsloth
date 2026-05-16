// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { toast } from "sonner";
import {
  DEFAULT_INFERENCE_PARAMS,
  type ChatLoraSummary,
  type ChatModelSummary,
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

export type ReasoningStyle = "enable_thinking" | "reasoning_effort";
export type ReasoningEffort =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "max"
  | "xhigh";

function loadReasoningEffort(fallback: ReasoningEffort): ReasoningEffort {
  if (!canUseStorage()) return fallback;
  try {
    const raw = localStorage.getItem(REASONING_EFFORT_KEY);
    if (
      raw === "none" ||
      raw === "minimal" ||
      raw === "low" ||
      raw === "medium" ||
      raw === "high" ||
      raw === "max" ||
      raw === "xhigh"
    ) {
      return raw;
    }
    return fallback;
  } catch {
    return fallback;
  }
}
let hasShownInferencePersistenceWarning = false;

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

function saveBool(key: string, value: boolean): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(key, value ? "true" : "false");
  } catch {
    // ignore
  }
}

function loadInt(key: string, fallback: number): number {
  if (!canUseStorage()) return fallback;
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    const parsed = parseInt(raw, 10);
    return Number.isNaN(parsed) ? fallback : parsed;
  } catch {
    return fallback;
  }
}

function saveInt(key: string, value: number): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(key, String(value));
  } catch {
    // ignore
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

function saveString(key: string, value: string): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(key, value);
  } catch {
    // ignore
  }
}

function asFiniteNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
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
      temperature: asFiniteNumber(parsed.temperature, DEFAULT_INFERENCE_PARAMS.temperature),
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
      maxTokens: asFiniteNumber(parsed.maxTokens, DEFAULT_INFERENCE_PARAMS.maxTokens),
      systemPrompt: asString(parsed.systemPrompt, DEFAULT_INFERENCE_PARAMS.systemPrompt),
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
  setLastOpenRouterChosenModel: (chosen: string | null) => void;
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
  supportsReasoningOff: false,
  reasoningEffortLevels: ["low", "medium", "high"],
  lastOpenRouterChosenModel: null,
  supportsPreserveThinking: false,
  preserveThinking: loadBool(PRESERVE_THINKING_KEY, false),
  supportsTools: false,
  supportsBuiltinWebSearch: false,
  supportsBuiltinCodeExecution: false,
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
  setModelLoading: (loading) => set({ modelLoading: loading }),
  setModelRequiresTrustRemoteCode: (modelRequiresTrustRemoteCode) =>
    set({ modelRequiresTrustRemoteCode }),
  setParams: (params) =>
    set(() => {
      const persisted = saveInferenceParams(params);
      if (!persisted && !hasShownInferencePersistenceWarning) {
        hasShownInferencePersistenceWarning = true;
        toast.warning("Chat settings could not be persisted", {
          description:
            "Your changes apply now, but may reset after refresh.",
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
      saveBool(AUTO_TITLE_KEY, autoTitle);
      return { autoTitle };
    }),
  setHfToken: (hfToken) =>
    set(() => {
      saveString(HF_TOKEN_KEY, hfToken);
      return { hfToken };
    }),
  setModelsError: (modelsError) => set({ modelsError }),
  setCheckpoint: (modelId, ggufVariant) =>
    set((state) => ({
      params: {
        ...state.params,
        checkpoint: modelId,
      },
      activeGgufVariant: ggufVariant ?? null,
    })),
  setActiveThreadId: (activeThreadId) => set({ activeThreadId, contextUsage: null }),
  setSettingsPanelOpen: (settingsPanelOpen) => set({ settingsPanelOpen }),
  clearCheckpoint: () =>
    set((state) => ({
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
    })),
  setReasoningEnabled: (reasoningEnabled) => set({ reasoningEnabled }),
  setLastOpenRouterChosenModel: (lastOpenRouterChosenModel) =>
    set({ lastOpenRouterChosenModel }),
  setReasoningStyle: (reasoningStyle) => set({ reasoningStyle }),
  setReasoningEffort: (reasoningEffort) =>
    set(() => {
      if (canUseStorage()) {
        try {
          localStorage.setItem(REASONING_EFFORT_KEY, reasoningEffort);
        } catch {
          // ignore
        }
      }
      return { reasoningEffort };
    }),
  setPreserveThinking: (preserveThinking) =>
    set(() => {
      saveBool(PRESERVE_THINKING_KEY, preserveThinking);
      return { preserveThinking };
    }),
  setToolsEnabled: (toolsEnabled) => set({ toolsEnabled }),
  setCodeToolsEnabled: (codeToolsEnabled) => set({ codeToolsEnabled }),
  setToolStatus: (toolStatus) => set({ toolStatus }),
  setGeneratingStatus: (generatingStatus) => set({ generatingStatus }),
  setAutoHealToolCalls: (autoHealToolCalls) =>
    set(() => {
      saveBool(AUTO_HEAL_TOOL_CALLS_KEY, autoHealToolCalls);
      return { autoHealToolCalls };
    }),
  setMaxToolCallsPerMessage: (maxToolCallsPerMessage) =>
    set(() => {
      saveInt(MAX_TOOL_CALLS_KEY, maxToolCallsPerMessage);
      return { maxToolCallsPerMessage };
    }),
  setToolCallTimeout: (toolCallTimeout) =>
    set(() => {
      saveInt(TOOL_CALL_TIMEOUT_KEY, toolCallTimeout);
      return { toolCallTimeout };
    }),
  setKvCacheDtype: (kvCacheDtype) => set({ kvCacheDtype }),
  setSpeculativeType: (speculativeType) => set({ speculativeType }),
  setCustomContextLength: (customContextLength) => set({ customContextLength }),
  setChatTemplateOverride: (chatTemplateOverride) => set({ chatTemplateOverride }),
  setPendingAudio: (base64, name) =>
    set({ pendingAudioBase64: base64, pendingAudioName: name }),
  clearPendingAudio: () =>
    set({ pendingAudioBase64: null, pendingAudioName: null }),
  setContextUsage: (contextUsage) => set({ contextUsage }),
}));
