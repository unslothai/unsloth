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
  toolStatus: string | null;
  generatingStatus: string | null;
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
  setToolStatus: (status: string | null) => void;
  setGeneratingStatus: (status: string | null) => void;
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
  toolStatus: null,
  generatingStatus: null,
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
      toolStatus: null,
      kvCacheDtype: null,
      defaultChatTemplate: null,
      chatTemplateOverride: null,
    })),
  setReasoningEnabled: (reasoningEnabled) => set({ reasoningEnabled }),
  setToolsEnabled: (toolsEnabled) => set({ toolsEnabled }),
  setToolStatus: (toolStatus) => set({ toolStatus }),
  setGeneratingStatus: (generatingStatus) => set({ generatingStatus }),
  setKvCacheDtype: (kvCacheDtype) => set({ kvCacheDtype }),
  setChatTemplateOverride: (chatTemplateOverride) => set({ chatTemplateOverride }),
  setPendingAudio: (base64, name) =>
    set({ pendingAudioBase64: base64, pendingAudioName: name }),
  clearPendingAudio: () =>
    set({ pendingAudioBase64: null, pendingAudioName: null }),
}));
