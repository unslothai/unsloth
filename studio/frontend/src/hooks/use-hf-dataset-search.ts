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
  plainTags: string[];
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
  const tags = ds.tags ?? [];
  const taskCategories = tags
    .filter((t) => t.startsWith("task_categories:"))
    .map((t) => t.slice("task_categories:".length));
  const plainTags = tags.filter((t) => !t.includes(":"));
  return {
    id: ds.name,
    downloads: ds.downloads,
    likes: ds.likes,
    totalExamples: extractTotalExamples(card),
    sizeCategory: card?.size_categories?.[0],
    taskCategories,
    plainTags,
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

type DatasetRelevance = "incompatible" | "neutral" | "boosted";

const BOOSTED_TASK_CATEGORIES: Record<ModelType, Set<string>> = {
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

const INCOMPATIBLE_TASKS_ALL_MODELS = new Set([
  "text-to-3d",
  "image-to-3d",
  "robotics",
  "reinforcement-learning",
  "tabular-classification",
  "tabular-regression",
  "time-series-forecasting",
]);

const PRETRAINING_PLAIN_TAGS = new Set(["pretraining", "pre-training"]);
const OCR_PLAIN_TAGS = new Set(["ocr", "document-ocr"]);

const PRETRAINING_SIZE_CATEGORIES = new Set([
  "100M<n<1B",
  "1B<n<10B",
  "10B<n<100B",
  "100B<n<1T",
  "n>1T",
]);

const OCR_OR_VISION_TEXT_TASKS = new Set([
  "image-to-text",
  "image-captioning",
  "visual-question-answering",
  "document-question-answering",
]);

const INCOMPATIBLE_TASKS_BY_MODEL: Record<ModelType, Set<string>> = {
  text: new Set([
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
    "visual-document-retrieval",
  ]),
  vision: new Set([
    "text-to-speech",
    "text-to-audio",
    "audio-classification",
    "audio-to-audio",
    "automatic-speech-recognition",
  ]),
  tts: new Set([
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
    "visual-document-retrieval",
  ]),
  embeddings: new Set([
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
    "visual-document-retrieval",
  ]),
};

function isPretrainingDataset(dataset: HfDatasetResult): boolean {
  if (dataset.plainTags.some((t) => PRETRAINING_PLAIN_TAGS.has(t.toLowerCase())))
    return true;
  if (
    dataset.sizeCategory &&
    PRETRAINING_SIZE_CATEGORIES.has(dataset.sizeCategory)
  )
    return true;
  return false;
}

function rankDatasetRelevance(
  dataset: HfDatasetResult,
  modelType: ModelType,
): DatasetRelevance {
  if (isPretrainingDataset(dataset)) return "incompatible";

  // Keep OCR / vision-text corpora out of non-vision defaults.
  if (modelType !== "vision") {
    if (
      dataset.plainTags.some((t) => OCR_PLAIN_TAGS.has(t.toLowerCase())) ||
      dataset.taskCategories.some((t) => OCR_OR_VISION_TEXT_TASKS.has(t))
    ) {
      return "incompatible";
    }
  }

  const { taskCategories } = dataset;
  if (taskCategories.length === 0) return "neutral";

  const boosted = BOOSTED_TASK_CATEGORIES[modelType];
  const modelIncompat = INCOMPATIBLE_TASKS_BY_MODEL[modelType];

  if (taskCategories.some((t) => boosted.has(t))) return "boosted";
  if (
    taskCategories.every(
      (t) => INCOMPATIBLE_TASKS_ALL_MODELS.has(t) || modelIncompat.has(t),
    )
  )
    return "incompatible";
  return "neutral";
}

function isOcrOrVisionTextDataset(dataset: HfDatasetResult): boolean {
  return (
    dataset.plainTags.some((t) => OCR_PLAIN_TAGS.has(t.toLowerCase())) ||
    dataset.taskCategories.some((t) => OCR_OR_VISION_TEXT_TASKS.has(t))
  );
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
    const hideOcr = modelType !== "vision";
    const baseResults = hideOcr
      ? search.results.filter((ds) => !isOcrOrVisionTextDataset(ds))
      : search.results;

    if (!modelType) return baseResults;

    const boosted: HfDatasetResult[] = [];
    const neutral: HfDatasetResult[] = [];

    for (const ds of baseResults) {
      const relevance = rankDatasetRelevance(ds, modelType);
      if (relevance === "boosted") boosted.push(ds);
      else if (relevance !== "incompatible") neutral.push(ds);
    }

    return [...boosted, ...neutral];
  }, [search.results, modelType]);

  return { ...search, results };
}
