import { create } from "zustand";
import {
  DEFAULT_INFERENCE_PARAMS,
  type ChatLoraSummary,
  type ChatModelSummary,
  type InferenceParams,
} from "../types/runtime";

type ChatRuntimeStore = {
  params: InferenceParams;
  models: ChatModelSummary[];
  loras: ChatLoraSummary[];
  warmingByThreadId: Record<string, boolean>;
  modelsError: string | null;
  setParams: (params: InferenceParams) => void;
  setModels: (models: ChatModelSummary[]) => void;
  setLoras: (loras: ChatLoraSummary[]) => void;
  setThreadWarming: (threadId: string, warming: boolean) => void;
  setModelsError: (error: string | null) => void;
  setCheckpoint: (modelId: string) => void;
  clearCheckpoint: () => void;
};

export const useChatRuntimeStore = create<ChatRuntimeStore>((set) => ({
  params: DEFAULT_INFERENCE_PARAMS,
  models: [],
  loras: [],
  warmingByThreadId: {},
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
