import { create } from "zustand";
import {
  DEFAULT_INFERENCE_PARAMS,
  type ChatModelSummary,
  type InferenceParams,
} from "../types/runtime";

type ChatRuntimeStore = {
  params: InferenceParams;
  models: ChatModelSummary[];
  modelsError: string | null;
  setParams: (params: InferenceParams) => void;
  setModels: (models: ChatModelSummary[]) => void;
  setModelsError: (error: string | null) => void;
  setCheckpoint: (modelId: string) => void;
  clearCheckpoint: () => void;
};

export const useChatRuntimeStore = create<ChatRuntimeStore>((set) => ({
  params: DEFAULT_INFERENCE_PARAMS,
  models: [],
  modelsError: null,
  setParams: (params) => set({ params }),
  setModels: (models) => set({ models }),
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
