import { useCallback } from "react";
import { toast } from "sonner";
import {
  getInferenceStatus,
  listLoras,
  listModels,
  loadModel,
  unloadModel,
} from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ChatLoraSummary, ChatModelSummary } from "../types/runtime";

const DEFAULT_MODEL_MAX_SEQ_LENGTH = 2048;

type SelectedModelInput = {
  id: string;
  isLora?: boolean;
};

const LORA_SUFFIX_RE = /_(\d{9,})$/;

function parseTrailingEpoch(input: string): number | undefined {
  const match = input.match(LORA_SUFFIX_RE);
  if (!match) {
    return undefined;
  }
  const parsed = Number.parseInt(match[1], 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function stripTrailingEpoch(input: string): string {
  const cleaned = input.replace(LORA_SUFFIX_RE, "").replace(/[_-]+$/, "").trim();
  return cleaned || input;
}

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

function toLoraSummary(lora: {
  display_name: string;
  adapter_path: string;
  base_model?: string | null;
}): ChatLoraSummary {
  const idTail = lora.adapter_path.split("/").filter(Boolean).at(-1) ?? "";
  const updatedAt =
    parseTrailingEpoch(lora.display_name) ?? parseTrailingEpoch(idTail);

  return {
    id: lora.adapter_path,
    name: stripTrailingEpoch(lora.display_name),
    baseModel: lora.base_model || "Unknown base model",
    updatedAt,
  };
}

export function useChatModelRuntime() {
  const params = useChatRuntimeStore((state) => state.params);
  const models = useChatRuntimeStore((state) => state.models);
  const loras = useChatRuntimeStore((state) => state.loras);
  const setModels = useChatRuntimeStore((state) => state.setModels);
  const setLoras = useChatRuntimeStore((state) => state.setLoras);
  const setModelsError = useChatRuntimeStore((state) => state.setModelsError);
  const setCheckpoint = useChatRuntimeStore((state) => state.setCheckpoint);
  const clearCheckpoint = useChatRuntimeStore((state) => state.clearCheckpoint);

  const refresh = useCallback(async () => {
    setModelsError(null);
    try {
      const [listRes, statusRes, lorasRes] = await Promise.all([
        listModels(),
        getInferenceStatus(),
        listLoras(),
      ]);

      setModels(listRes.models.map(toChatModelSummary));
      setLoras(lorasRes.loras.map(toLoraSummary));

      if (statusRes.active_model) {
        setCheckpoint(statusRes.active_model);
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to load models";
      setModelsError(message);
    }
  }, [setCheckpoint, setLoras, setModels, setModelsError]);

  const selectModel = useCallback(
    async (selection: string | SelectedModelInput) => {
      const modelId = typeof selection === "string" ? selection : selection.id;
      if (!modelId || params.checkpoint === modelId) {
        return;
      }

      const explicitIsLora =
        typeof selection === "string" ? undefined : selection.isLora;
      const model = models.find((entry) => entry.id === modelId);
      const lora = loras.find((entry) => entry.id === modelId);
      const isLora =
        explicitIsLora ?? model?.isLora ?? (lora ? true : false);
      const displayName = model?.name || lora?.name || modelId;
      const loadingToastId = toast.loading(`Loading ${displayName}...`);

      setModelsError(null);
      try {
        if (params.checkpoint) {
          await unloadModel({ model_path: params.checkpoint });
        }

        await loadModel({
          model_path: modelId,
          hf_token: null,
          max_seq_length: DEFAULT_MODEL_MAX_SEQ_LENGTH,
          load_in_4bit: true,
          is_lora: isLora,
        });

        setCheckpoint(modelId);
        await refresh();
        toast.success(`${displayName} loaded`, { id: loadingToastId });
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to load model";
        setModelsError(message);
        toast.error(message, { id: loadingToastId });
      }
    },
    [loras, models, params.checkpoint, refresh, setCheckpoint, setModelsError],
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
  }, [clearCheckpoint, params.checkpoint, refresh, setModelsError]);

  return {
    refresh,
    selectModel,
    ejectModel,
  };
}
