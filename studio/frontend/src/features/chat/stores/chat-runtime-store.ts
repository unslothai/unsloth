// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";
import { create } from "zustand";
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
import {
  loadChatSettingsWithLegacyImport,
  savePersistedChatSettingsPatch,
} from "../utils/chat-settings-storage";

const HF_TOKEN_KEY = "unsloth_hf_token";

export type ReasoningStyle = "enable_thinking" | "reasoning_effort";
export type ReasoningEffort =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "max"
  | "xhigh";

let hasShownSettingsPersistenceWarning = false;
let settingsMutationVersion = 0;
let settingsSaveQueue: Promise<void> = Promise.resolve();

function warnSettingsPersistenceFailure(): void {
  if (hasShownSettingsPersistenceWarning) {
    return;
  }
  hasShownSettingsPersistenceWarning = true;
  toast.warning("Chat settings could not be persisted", {
    description: "Your changes apply now, but may reset after refresh.",
  });
}

function saveSettingsPatch(
  patch: Parameters<typeof savePersistedChatSettingsPatch>[0],
): void {
  settingsMutationVersion += 1;
  settingsSaveQueue = settingsSaveQueue
    .catch(() => undefined)
    .then(async () => {
      try {
        await savePersistedChatSettingsPatch(patch);
      } catch {
        warnSettingsPersistenceFailure();
      }
    });
}

function canUseStorage(): boolean {
  return typeof window !== "undefined";
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

export const useChatRuntimeStore = create<ChatRuntimeStore>((set, get) => ({
  settingsHydrated: false,
  params: DEFAULT_INFERENCE_PARAMS,
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
  reasoningEnabled: true,
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
  toolsEnabled: false,
  codeToolsEnabled: false,
  toolStatus: null,
  generatingStatus: null,
  autoHealToolCalls: true,
  maxToolCallsPerMessage: 25,
  toolCallTimeout: 5,
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
  hydratePersistedSettings: async () => {
    if (get().settingsHydrated) {
      return;
    }
    const hydrationVersion = settingsMutationVersion;
    try {
      const settings = await loadChatSettingsWithLegacyImport();
      set((state) => {
        if (state.settingsHydrated) {
          return state;
        }
        if (settingsMutationVersion !== hydrationVersion) {
          return { settingsHydrated: true };
        }
        const activePreset = settings.activePreset ?? state.activePreset;
        const customPresets =
          settings.customPresets?.map((preset) => ({
            name: preset.name,
            params: {
              ...DEFAULT_INFERENCE_PARAMS,
              ...preset.params,
            },
          })) ?? state.customPresets;
        const params = {
          ...state.params,
          ...settings.inferenceParams,
          checkpoint: state.params.checkpoint,
        };
        return {
          settingsHydrated: true,
          params,
          customPresets,
          activePreset,
          activePresetSource:
            settings.activePresetSource ?? getPresetSource(activePreset),
          autoTitle: settings.autoTitle ?? state.autoTitle,
          reasoningEffort: settings.reasoningEffort ?? state.reasoningEffort,
          preserveThinking: settings.preserveThinking ?? state.preserveThinking,
          autoHealToolCalls:
            settings.autoHealToolCalls ?? state.autoHealToolCalls,
          maxToolCallsPerMessage:
            settings.maxToolCallsPerMessage ?? state.maxToolCallsPerMessage,
          toolCallTimeout: settings.toolCallTimeout ?? state.toolCallTimeout,
        };
      });
    } catch {
      set((state) =>
        state.settingsHydrated ? state : { settingsHydrated: true },
      );
      warnSettingsPersistenceFailure();
    }
  },
  setModelLoading: (loading) => set({ modelLoading: loading }),
  setModelRequiresTrustRemoteCode: (modelRequiresTrustRemoteCode) =>
    set({ modelRequiresTrustRemoteCode }),
  setParams: (params) =>
    set(() => {
      const { checkpoint: _checkpoint, ...persistedParams } = params;
      saveSettingsPatch({ inferenceParams: persistedParams });
      return { params };
    }),
  setCustomPresets: (customPresets) =>
    set(() => {
      saveSettingsPatch({ customPresets });
      return { customPresets };
    }),
  setActivePreset: (activePreset) =>
    set(() => {
      saveSettingsPatch({ activePreset });
      return { activePreset };
    }),
  setActivePresetSource: (activePresetSource) =>
    set(() => {
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
    set(() => {
      saveSettingsPatch({ autoTitle });
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
  setActiveThreadId: (activeThreadId) =>
    set({ activeThreadId, contextUsage: null }),
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
      saveSettingsPatch({ reasoningEffort });
      return { reasoningEffort };
    }),
  setPreserveThinking: (preserveThinking) =>
    set(() => {
      saveSettingsPatch({ preserveThinking });
      return { preserveThinking };
    }),
  setToolsEnabled: (toolsEnabled) => set({ toolsEnabled }),
  setCodeToolsEnabled: (codeToolsEnabled) => set({ codeToolsEnabled }),
  setToolStatus: (toolStatus) => set({ toolStatus }),
  setGeneratingStatus: (generatingStatus) => set({ generatingStatus }),
  setAutoHealToolCalls: (autoHealToolCalls) =>
    set(() => {
      saveSettingsPatch({ autoHealToolCalls });
      return { autoHealToolCalls };
    }),
  setMaxToolCallsPerMessage: (maxToolCallsPerMessage) =>
    set(() => {
      saveSettingsPatch({ maxToolCallsPerMessage });
      return { maxToolCallsPerMessage };
    }),
  setToolCallTimeout: (toolCallTimeout) =>
    set(() => {
      saveSettingsPatch({ toolCallTimeout });
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
