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

function mapModel(raw: unknown): HfModelResult {
  const m = raw as {
    name: string;
    downloads: number;
    likes: number;
    safetensors?: { total: number };
  };
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
        search: { query, ...(task ? { task } : {}) },
        additionalFields: ["safetensors"],
        ...(accessToken ? { credentials: { accessToken } } : {}),
      }) as AsyncGenerator<unknown>,
    [query, task, accessToken],
  );

  return useHfPaginatedSearch(query, createIter, mapModel);
}
