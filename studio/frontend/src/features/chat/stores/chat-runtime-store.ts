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
  warmingByThreadId: Record<string, boolean>;
  runningByThreadId: Record<string, boolean>;
  autoTitle: boolean;
  modelsError: string | null;
  setParams: (params: InferenceParams) => void;
  setModels: (models: ChatModelSummary[]) => void;
  setLoras: (loras: ChatLoraSummary[]) => void;
  setThreadWarming: (threadId: string, warming: boolean) => void;
  setThreadRunning: (threadId: string, running: boolean) => void;
  setAutoTitle: (enabled: boolean) => void;
  setModelsError: (error: string | null) => void;
  setCheckpoint: (modelId: string) => void;
  clearCheckpoint: () => void;
};

export const useChatRuntimeStore = create<ChatRuntimeStore>((set) => ({
  params: DEFAULT_INFERENCE_PARAMS,
  models: [],
  loras: [],
  warmingByThreadId: {},
  runningByThreadId: {},
  autoTitle: loadBool(AUTO_TITLE_KEY, false),
  modelsError: null,
  setParams: (params) => set({ params }),
  setModels: (models) => set({ models }),
  setLoras: (loras) => set({ loras }),
  setThreadWarming: (threadId, warming) =>
    set((state) => {
      const next = { ...state.warmingByThreadId };
      if (warming) {
        next[threadId] = true;
      } else {
        delete next[threadId];
      }
      return { warmingByThreadId: next };
    }),
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
  setCheckpoint: (modelId) =>
    set((state) => ({
      params: {
        ...state.params,
        checkpoint: modelId,
      },
    })),
  clearCheckpoint: () =>
    set((state) => ({
      params: {
        ...state.params,
        checkpoint: "",
      },
    })),
}));
