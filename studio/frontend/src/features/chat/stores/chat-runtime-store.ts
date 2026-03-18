// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  DEFAULT_INFERENCE_PARAMS,
  type ChatLoraSummary,
  type ChatModelSummary,
  type InferenceParams,
} from "../types/runtime";

const AUTO_TITLE_KEY = "unsloth_chat_auto_title";
const AUTO_HEAL_TOOL_CALLS_KEY = "unsloth_auto_heal_tool_calls";
const MAX_TOOL_CALLS_KEY = "unsloth_max_tool_calls_per_message";
const TOOL_CALL_TIMEOUT_KEY = "unsloth_tool_call_timeout";

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

type ChatRuntimeStore = {
  params: InferenceParams;
  models: ChatModelSummary[];
  loras: ChatLoraSummary[];
  runningByThreadId: Record<string, boolean>;
  autoTitle: boolean;
  modelsError: string | null;
  activeGgufVariant: string | null;
  ggufContextLength: number | null;
  supportsReasoning: boolean;
  reasoningEnabled: boolean;
  supportsTools: boolean;
  toolsEnabled: boolean;
  codeToolsEnabled: boolean;
  toolStatus: string | null;
  generatingStatus: string | null;
  autoHealToolCalls: boolean;
  maxToolCallsPerMessage: number;
  toolCallTimeout: number;
  kvCacheDtype: string | null;
  defaultChatTemplate: string | null;
  chatTemplateOverride: string | null;
  activeThreadId: string | null;
  pendingAudioBase64: string | null;
  pendingAudioName: string | null;
  modelLoading: boolean;
  setModelLoading: (loading: boolean) => void;
  setParams: (params: InferenceParams) => void;
  setModels: (models: ChatModelSummary[]) => void;
  setLoras: (loras: ChatLoraSummary[]) => void;
  setThreadRunning: (threadId: string, running: boolean) => void;
  setAutoTitle: (enabled: boolean) => void;
  setModelsError: (error: string | null) => void;
  setCheckpoint: (modelId: string, ggufVariant?: string | null) => void;
  setActiveThreadId: (threadId: string | null) => void;
  clearCheckpoint: () => void;
  setReasoningEnabled: (enabled: boolean) => void;
  setToolsEnabled: (enabled: boolean) => void;
  setCodeToolsEnabled: (enabled: boolean) => void;
  setToolStatus: (status: string | null) => void;
  setGeneratingStatus: (status: string | null) => void;
  setAutoHealToolCalls: (enabled: boolean) => void;
  setMaxToolCallsPerMessage: (value: number) => void;
  setToolCallTimeout: (value: number) => void;
  setKvCacheDtype: (dtype: string | null) => void;
  setChatTemplateOverride: (template: string | null) => void;
  setPendingAudio: (base64: string, name: string) => void;
  clearPendingAudio: () => void;
};

export const useChatRuntimeStore = create<ChatRuntimeStore>((set) => ({
  params: DEFAULT_INFERENCE_PARAMS,
  models: [],
  loras: [],
  runningByThreadId: {},
  autoTitle: loadBool(AUTO_TITLE_KEY, false),
  modelsError: null,
  activeGgufVariant: null,
  ggufContextLength: null,
  supportsReasoning: false,
  reasoningEnabled: true,
  supportsTools: false,
  toolsEnabled: false,
  codeToolsEnabled: false,
  toolStatus: null,
  generatingStatus: null,
  autoHealToolCalls: loadBool(AUTO_HEAL_TOOL_CALLS_KEY, true),
  maxToolCallsPerMessage: loadInt(MAX_TOOL_CALLS_KEY, 10),
  toolCallTimeout: loadInt(TOOL_CALL_TIMEOUT_KEY, 5),
  kvCacheDtype: null,
  defaultChatTemplate: null,
  chatTemplateOverride: null,
  activeThreadId: null,
  pendingAudioBase64: null,
  pendingAudioName: null,
  modelLoading: false,
  setModelLoading: (loading) => set({ modelLoading: loading }),
  setParams: (params) => set({ params }),
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
  setAutoTitle: (autoTitle) =>
    set(() => {
      saveBool(AUTO_TITLE_KEY, autoTitle);
      return { autoTitle };
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
  setActiveThreadId: (activeThreadId) => set({ activeThreadId }),
  clearCheckpoint: () =>
    set((state) => ({
      params: {
        ...state.params,
        checkpoint: "",
      },
      activeGgufVariant: null,
      ggufContextLength: null,
      supportsReasoning: false,
      reasoningEnabled: true,
      supportsTools: false,
      toolsEnabled: false,
      codeToolsEnabled: false,
      toolStatus: null,
      kvCacheDtype: null,
      defaultChatTemplate: null,
      chatTemplateOverride: null,
    })),
  setReasoningEnabled: (reasoningEnabled) => set({ reasoningEnabled }),
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
  setChatTemplateOverride: (chatTemplateOverride) => set({ chatTemplateOverride }),
  setPendingAudio: (base64, name) =>
    set({ pendingAudioBase64: base64, pendingAudioName: name }),
  clearPendingAudio: () =>
    set({ pendingAudioBase64: null, pendingAudioName: null }),
}));
