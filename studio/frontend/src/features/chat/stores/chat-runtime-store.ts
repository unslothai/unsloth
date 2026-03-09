// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

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
  pendingAudioBase64: string | null;
  pendingAudioName: string | null;
  setParams: (params: InferenceParams) => void;
  setModels: (models: ChatModelSummary[]) => void;
  setLoras: (loras: ChatLoraSummary[]) => void;
  setThreadRunning: (threadId: string, running: boolean) => void;
  setAutoTitle: (enabled: boolean) => void;
  setModelsError: (error: string | null) => void;
  setCheckpoint: (modelId: string, ggufVariant?: string | null) => void;
  clearCheckpoint: () => void;
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
  pendingAudioBase64: null,
  pendingAudioName: null,
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
  clearCheckpoint: () =>
    set((state) => ({
      params: {
        ...state.params,
        checkpoint: "",
      },
      activeGgufVariant: null,
    })),
  setPendingAudio: (base64, name) =>
    set({ pendingAudioBase64: base64, pendingAudioName: name }),
  clearPendingAudio: () =>
    set({ pendingAudioBase64: null, pendingAudioName: null }),
}));
