import { listDatasets } from "@huggingface/hub";
import { useCallback } from "react";
import { useHfPaginatedSearch } from "./use-hf-paginated-search";

interface DatasetInfoSplit {
  name: string;
  num_bytes: number;
  num_examples: number;
}

interface CardDataWithInfo {
  size_categories?: string[];
  pretty_name?: string;
  dataset_info?:
    | {
        splits?: DatasetInfoSplit[];
        download_size?: number;
        dataset_size?: number;
      }
    | Array<{ splits?: DatasetInfoSplit[] }>;
}

function extractTotalExamples(
  cardData: CardDataWithInfo | undefined,
): number | undefined {
  if (!cardData?.dataset_info) {
    return undefined;
  }
  const infos = Array.isArray(cardData.dataset_info)
    ? cardData.dataset_info
    : [cardData.dataset_info];
  let total = 0;
  let found = false;
  for (const info of infos) {
    for (const split of info.splits ?? []) {
      if (typeof split.num_examples === "number") {
        total += split.num_examples;
        found = true;
      }
    }
  }
  return found ? total : undefined;
}

export interface HfDatasetResult {
  id: string;
  downloads: number;
  likes: number;
  totalExamples?: number;
  sizeCategory?: string;
}

function mapDataset(raw: unknown): HfDatasetResult {
  const ds = raw as {
    name: string;
    downloads: number;
    likes: number;
    cardData?: unknown;
  };
  const card = ds.cardData as CardDataWithInfo | undefined;
  return {
    id: ds.name,
    downloads: ds.downloads,
    likes: ds.likes,
    totalExamples: extractTotalExamples(card),
    sizeCategory: card?.size_categories?.[0],
  };
}

export function useHfDatasetSearch(
  query: string,
  options?: { accessToken?: string },
) {
  const { accessToken } = options ?? {};

  const createIter = useCallback(
    () =>
      listDatasets({
        search: { query },
        additionalFields: ["cardData"],
        ...(accessToken ? { credentials: { accessToken } } : {}),
      }) as AsyncGenerator<unknown>,
    [query, accessToken],
  );

  return useHfPaginatedSearch(query, createIter, mapDataset);
}
