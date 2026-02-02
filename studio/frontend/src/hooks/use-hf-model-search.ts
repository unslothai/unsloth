import type { PipelineType } from "@huggingface/hub";
import { listModels } from "@huggingface/hub";
import { useCallback } from "react";
import { useHfPaginatedSearch } from "./use-hf-paginated-search";

export interface HfModelResult {
  id: string;
  downloads: number;
  likes: number;
  totalParams?: number;
}

const EXCLUDED_TAGS = new Set([
  "gguf",
  "gptq",
  "awq",
  "exl2",
  "mlx",
  "onnx",
  "openvino",
  "coreml",
  "tflite",
  "ctranslate2",
]);

function mapModel(raw: unknown): HfModelResult | null {
  const m = raw as {
    name: string;
    downloads: number;
    likes: number;
    safetensors?: { total: number };
    tags?: string[];
  };
  if (m.tags?.some((t) => EXCLUDED_TAGS.has(t))) {
    return null;
  }
  return {
    id: m.name,
    downloads: m.downloads,
    likes: m.likes,
    totalParams: m.safetensors?.total,
  };
}

export function useHfModelSearch(
  query: string,
  options?: { task?: PipelineType; accessToken?: string },
) {
  const { task, accessToken } = options ?? {};

  const createIter = useCallback(
    () =>
      listModels({
        search: {
          ...(query.trim() ? { query } : {}),
          tags: ["transformers"],
          ...(task ? { task } : {}),
        },
        additionalFields: ["safetensors", "tags"],
        ...(accessToken ? { credentials: { accessToken } } : {}),
      }) as AsyncGenerator<unknown>,
    [query, task, accessToken],
  );

  return useHfPaginatedSearch(createIter, mapModel);
}
