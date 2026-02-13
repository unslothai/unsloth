import { useCallback } from "react";
import {
  getInferenceStatus,
  listModels,
  loadModel,
  unloadModel,
} from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ChatModelSummary } from "../types/runtime";

const DEFAULT_MODEL_MAX_SEQ_LENGTH = 2048;

function describeModel(model: {
  is_lora?: boolean;
  is_vision?: boolean;
}): string | undefined {
  const tags: string[] = [];
  if (model.is_lora) tags.push("LoRA");
  if (model.is_vision) tags.push("Vision");
  if (!model.is_lora && !model.is_vision) tags.push("Base");
  return tags.join(" · ");
}

function toChatModelSummary(model: {
  id: string;
  name?: string | null;
  is_lora?: boolean;
  is_vision?: boolean;
}): ChatModelSummary {
  return {
    id: model.id,
    name: model.name || model.id,
    description: describeModel(model),
    isLora: Boolean(model.is_lora),
    isVision: Boolean(model.is_vision),
  };
}

export function useChatModelRuntime() {
  const params = useChatRuntimeStore((state) => state.params);
  const models = useChatRuntimeStore((state) => state.models);
  const setModels = useChatRuntimeStore((state) => state.setModels);
  const setModelsError = useChatRuntimeStore((state) => state.setModelsError);
  const setCheckpoint = useChatRuntimeStore((state) => state.setCheckpoint);
  const clearCheckpoint = useChatRuntimeStore((state) => state.clearCheckpoint);

  const refresh = useCallback(async () => {
    setModelsError(null);
    try {
      const [listRes, statusRes] = await Promise.all([
        listModels(),
        getInferenceStatus(),
      ]);

      const modelList = listRes.models.map(toChatModelSummary);
      setModels(modelList);

      if (statusRes.active_model) {
        setCheckpoint(statusRes.active_model);
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to load models";
      setModelsError(message);
    }
  }, [
    setCheckpoint,
    setModels,
    setModelsError,
  ]);

  const selectModel = useCallback(
    async (modelId: string) => {
      if (!modelId || params.checkpoint === modelId) {
        return;
      }
      const selected = models.find((model) => model.id === modelId);
      if (!selected) {
        setModelsError("Selected model was not found in model list.");
        return;
      }

      setModelsError(null);
      try {
        if (params.checkpoint) {
          await unloadModel({ model_path: params.checkpoint });
        }

        await loadModel({
          model_path: selected.id,
          hf_token: null,
          max_seq_length: DEFAULT_MODEL_MAX_SEQ_LENGTH,
          load_in_4bit: true,
          is_lora: selected.isLora,
        });

        setCheckpoint(selected.id);
        await refresh();
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to load model";
        setModelsError(message);
      }
    },
    [
      models,
      params.checkpoint,
      refresh,
      setCheckpoint,
      setModelsError,
    ],
  );

  const ejectModel = useCallback(async () => {
    if (!params.checkpoint) {
      return;
    }
    setModelsError(null);
    try {
      await unloadModel({ model_path: params.checkpoint });
      clearCheckpoint();
      await refresh();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to unload model";
      setModelsError(message);
    }
  }, [
    clearCheckpoint,
    params.checkpoint,
    refresh,
    setModelsError,
  ]);

  return {
    refresh,
    selectModel,
    ejectModel,
  };
}
