import { listDatasets } from "@huggingface/hub";
import { useCallback, useMemo } from "react";
import type { ModelType } from "@/types/training";
import { useHfPaginatedSearch } from "./use-hf-paginated-search";

interface DatasetInfoSplit {
  name: string;
  // biome-ignore lint/style/useNamingConvention: external schema
  num_bytes: number;
  // biome-ignore lint/style/useNamingConvention: external schema
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

  const examples = infos
    .flatMap((info) => info.splits ?? [])
    .filter((s) => typeof s.num_examples === "number")
    .map((s) => s.num_examples);

  return examples.length > 0 ? examples.reduce((a, b) => a + b, 0) : undefined;
}

export interface HfDatasetResult {
  id: string;
  downloads: number;
  likes: number;
  totalExamples?: number;
  sizeCategory?: string;
  taskCategories: string[];
}

function mapDataset(raw: unknown): HfDatasetResult {
  const ds = raw as {
    name: string;
    downloads: number;
    likes: number;
    tags?: string[];
    cardData?: unknown;
  };
  const card = ds.cardData as CardDataWithInfo | undefined;
  const taskCategories = (ds.tags ?? [])
    .filter((t) => t.startsWith("task_categories:"))
    .map((t) => t.slice("task_categories:".length));
  return {
    id: ds.name,
    downloads: ds.downloads,
    likes: ds.likes,
    totalExamples: extractTotalExamples(card),
    sizeCategory: card?.size_categories?.[0],
    taskCategories,
  };
}

function withTrendingSort(
  input: Parameters<typeof fetch>[0],
  init?: Parameters<typeof fetch>[1],
): ReturnType<typeof fetch> {
  const rawUrl =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;
  const url = new URL(rawUrl);

  if (!url.searchParams.has("sort")) {
    url.searchParams.set("sort", "trendingScore");
  }
  if (!url.searchParams.has("direction")) {
    url.searchParams.set("direction", "-1");
  }

  return fetch(url, init);
}

const RELEVANT_TASK_CATEGORIES: Record<ModelType, Set<string>> = {
  text: new Set([
    "text-generation",
    "text2text-generation",
    "question-answering",
    "summarization",
    "conversational",
  ]),
  vision: new Set([
    "image-text-to-text",
    "visual-question-answering",
    "image-to-text",
    "image-captioning",
  ]),
  tts: new Set([
    "text-to-speech",
    "text-to-audio",
    "automatic-speech-recognition",
  ]),
  embeddings: new Set([
    "feature-extraction",
    "sentence-similarity",
    "text-retrieval",
  ]),
};

const INCOMPATIBLE_TASK_CATEGORIES: Record<ModelType, Set<string>> = {
  text: new Set([
    "text-to-3d",
    "image-to-3d",
    "text-to-image",
    "image-to-image",
    "image-to-video",
    "text-to-video",
    "image-classification",
    "image-feature-extraction",
    "image-text-to-image",
    "zero-shot-image-classification",
    "keypoint-detection",
    "object-detection",
    "image-segmentation",
    "depth-estimation",
    "text-to-speech",
    "text-to-audio",
    "audio-classification",
    "audio-to-audio",
    "automatic-speech-recognition",
    "video-classification",
    "robotics",
    "reinforcement-learning",
    "tabular-classification",
    "tabular-regression",
    "time-series-forecasting",
    "visual-document-retrieval",
  ]),
  vision: new Set([
    "text-to-3d",
    "image-to-3d",
    "text-to-speech",
    "text-to-audio",
    "audio-classification",
    "audio-to-audio",
    "automatic-speech-recognition",
    "robotics",
    "reinforcement-learning",
    "tabular-classification",
    "tabular-regression",
    "time-series-forecasting",
  ]),
  tts: new Set([
    "text-to-3d",
    "image-to-3d",
    "text-to-image",
    "image-to-image",
    "image-to-video",
    "text-to-video",
    "image-classification",
    "image-feature-extraction",
    "image-text-to-image",
    "zero-shot-image-classification",
    "keypoint-detection",
    "object-detection",
    "image-segmentation",
    "depth-estimation",
    "video-classification",
    "robotics",
    "reinforcement-learning",
    "tabular-classification",
    "tabular-regression",
    "time-series-forecasting",
    "visual-document-retrieval",
  ]),
  embeddings: new Set([
    "text-to-3d",
    "image-to-3d",
    "text-to-image",
    "image-to-image",
    "image-to-video",
    "text-to-video",
    "image-classification",
    "image-feature-extraction",
    "image-text-to-image",
    "zero-shot-image-classification",
    "keypoint-detection",
    "object-detection",
    "image-segmentation",
    "depth-estimation",
    "text-to-speech",
    "text-to-audio",
    "audio-classification",
    "audio-to-audio",
    "automatic-speech-recognition",
    "video-classification",
    "robotics",
    "reinforcement-learning",
    "tabular-classification",
    "tabular-regression",
    "time-series-forecasting",
    "visual-document-retrieval",
  ]),
};

function classifyDataset(
  dataset: HfDatasetResult,
  modelType: ModelType,
): -1 | 0 | 1 {
  const { taskCategories } = dataset;
  if (taskCategories.length === 0) return 0;

  const relevant = RELEVANT_TASK_CATEGORIES[modelType];
  const incompatible = INCOMPATIBLE_TASK_CATEGORIES[modelType];

  if (taskCategories.some((t) => relevant.has(t))) return 1;
  if (taskCategories.every((t) => incompatible.has(t))) return -1;
  return 0;
}

export function useHfDatasetSearch(
  query: string,
  options?: { modelType?: ModelType | null; accessToken?: string },
) {
  const { modelType, accessToken } = options ?? {};
  const createIter = useCallback(
    () =>
      listDatasets({
        search: query.trim() ? { query } : {},
        additionalFields: ["cardData", "tags"],
        fetch: withTrendingSort,
        ...(accessToken ? { credentials: { accessToken } } : {}),
      }) as AsyncGenerator<unknown>,
    [query, accessToken],
  );

  const search = useHfPaginatedSearch(createIter, mapDataset);

  const results = useMemo(() => {
    if (!modelType) return search.results;

    const boosted: HfDatasetResult[] = [];
    const neutral: HfDatasetResult[] = [];

    for (const ds of search.results) {
      const rank = classifyDataset(ds, modelType);
      if (rank === 1) boosted.push(ds);
      else if (rank !== -1) neutral.push(ds);
    }

    return [...boosted, ...neutral];
  }, [search.results, modelType]);

  return { ...search, results };
}
