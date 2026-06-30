// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fetchWithTimeout } from "../lib/network";
import type { HubModelType } from "../types";
import { listDatasets } from "@huggingface/hub";
import { useCallback, useMemo } from "react";
import { useHubPaginatedSearch } from "./use-hub-paginated-search";

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
  private?: boolean;
  gated?: false | "auto" | "manual";
  updatedAt?: string;
  createdAt?: string;
  downloadsAllTime?: number;
  prettyName?: string;
  totalExamples?: number;
  sizeCategory?: string;
  taskCategories: string[];
  plainTags: string[];
}

function mapDataset(raw: unknown): HfDatasetResult {
  const ds = raw as {
    name: string;
    downloads?: number;
    downloadsAllTime?: number;
    likes?: number;
    private?: boolean;
    gated?: false | "auto" | "manual";
    updatedAt?: Date | string;
    createdAt?: Date | string;
    tags?: string[];
    cardData?: unknown;
  };
  const card = ds.cardData as CardDataWithInfo | undefined;
  const tags = ds.tags ?? [];
  const taskCategories = tags
    .filter((t) => t.startsWith("task_categories:"))
    .map((t) => t.slice("task_categories:".length));
  const plainTags = tags.filter((t) => !t.includes(":"));
  const updatedAt =
    ds.updatedAt instanceof Date
      ? ds.updatedAt.toISOString()
      : typeof ds.updatedAt === "string"
        ? ds.updatedAt
        : undefined;
  const createdAt =
    ds.createdAt instanceof Date
      ? ds.createdAt.toISOString()
      : typeof ds.createdAt === "string"
        ? ds.createdAt
        : undefined;
  return {
    id: ds.name,
    downloads: ds.downloads ?? 0,
    likes: ds.likes ?? 0,
    private: ds.private,
    gated: ds.gated,
    updatedAt,
    createdAt,
    downloadsAllTime: ds.downloadsAllTime,
    prettyName: card?.pretty_name,
    totalExamples: extractTotalExamples(card),
    sizeCategory: card?.size_categories?.[0],
    taskCategories,
    plainTags,
  };
}

type DatasetSortKey =
  | "trendingScore"
  | "downloads"
  | "likes"
  | "lastModified"
  | "createdAt";
type DatasetSortDirection = "desc" | "asc";

const DESC_ONLY_DATASET_SORTS = new Set<DatasetSortKey>(["trendingScore"]);
const HF_DATASET_SEARCH_TIMEOUT_MS = 15_000;

function makeDatasetSortFetch(
  sortBy: DatasetSortKey,
  direction: DatasetSortDirection,
  signal?: AbortSignal,
): typeof fetch {
  return (input, init) => {
    const rawUrl =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
    const url = new URL(rawUrl);

    if (!url.searchParams.has("sort")) {
      url.searchParams.set("sort", sortBy);
    }
    const effectiveSort = (url.searchParams.get("sort") ?? sortBy) as
      | DatasetSortKey
      | undefined;
    const effectiveDir =
      effectiveSort && DESC_ONLY_DATASET_SORTS.has(effectiveSort)
        ? "desc"
        : direction;
    url.searchParams.set("direction", effectiveDir === "asc" ? "1" : "-1");

    return fetchWithTimeout(
      url,
      signal ? { ...init, signal } : init,
      HF_DATASET_SEARCH_TIMEOUT_MS,
    );
  };
}

type DatasetRelevance = "incompatible" | "neutral" | "boosted";

const BOOSTED_TASK_CATEGORIES: Record<HubModelType, Set<string>> = {
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
  audio: new Set([
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
  "5M<n<10M",
  "10M<n<100M",
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

const CURATED_EMPTY_QUERY_DATASET_IDS: Partial<Record<HubModelType, string[]>> = {
  text: [
    "unsloth/alpaca-cleaned",
    "unsloth/OpenMathReasoning-mini",
    "mlabonne/FineTome-100k",
    "openai/gsm8k",
    "philschmid/guanaco-sharegpt-style",
    "open-r1/DAPO-Math-17k-Processed",
    "HuggingFaceH4/Multilingual-Thinking",
    "HuggingFaceH4/ultrafeedback_binarized",
    "reciperesearch/dolphin-sft-v0.1-preference",
    "roneneldan/TinyStories",
    "FreedomIntelligence/alpaca-gpt4-korean",
    "Goedel-LM/SFT_dataset_v2",
    "allenai/tulu-3-sft-mixture",
    "HuggingFaceH4/no_robots",
    "Magpie-Align/Magpie-Air-300K-Filtered",
    "teknium/OpenHermes-2.5",
    "databricks/databricks-dolly-15k",
    "tatsu-lab/alpaca",
    "garage-bAInd/Open-Platypus",
    "microsoft/orca-math-word-problems-200k",
    "Open-Orca/OpenOrca",
    "openbmb/UltraInteract_sft",
  ],
  vision: [
    "unsloth/LaTeX_OCR",
    "unsloth/llava-instruct-mix-vsft-mini",
    "unsloth/Radiology_mini",
    "AI4Math/MathVista",
    "AI4Math/MathVerse",
    "ChongyanChen/VQAonline",
    "lmms-lab/VQAv2",
    "hezarai/parsynth-ocr-200k",
  ],
  audio: [
    "MrDragonFox/Elise",
    "keithito/lj_speech",
    "parler-tts/mls_eng_10k",
    "parler-tts/libritts-r-filtered-speaker-descriptions",
    "openslr/librispeech_asr",
    "MikhailT/hifi-tts",
    "mozilla-foundation/common_voice_17_0",
    "facebook/voxpopuli",
    "speechcolab/gigaspeech",
    "kth-tmh/vctk",
    "Wenetspeech4TTS/WenetSpeech4TTS",
  ],
  embeddings: [
    "electroglyph/technical",
  ],
};

const INCOMPATIBLE_TASKS_BY_MODEL: Record<HubModelType, Set<string>> = {
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
  audio: new Set([
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
  modelType: HubModelType,
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

function toCuratedDatasetResult(id: string): HfDatasetResult {
  // Curated defaults are id-only; this fills the shared result shape without extra HF requests.
  return {
    id,
    downloads: 0,
    likes: 0,
    taskCategories: [],
    plainTags: [],
  };
}

export function useHubDatasetSearch(
  query: string,
  options?: {
    modelType?: HubModelType | null;
    accessToken?: string;
    enabled?: boolean;
    sortBy?: DatasetSortKey;
    sortDirection?: DatasetSortDirection;
  },
) {
  const {
    modelType,
    accessToken,
    enabled = true,
    sortBy = "trendingScore",
    sortDirection = "desc",
  } = options ?? {};
  const hasQuery = query.trim().length > 0;
  const useCuratedOnly = !hasQuery && !!modelType;
  const createIter = useCallback(
    (signal: AbortSignal) => {
      if (useCuratedOnly) {
        return (async function* empty() {})() as AsyncGenerator<unknown>;
      }
      return listDatasets({
        search: hasQuery ? { query } : {},
        additionalFields: ["cardData", "tags", "createdAt", "downloadsAllTime"],
        fetch: makeDatasetSortFetch(sortBy, sortDirection, signal),
        ...(accessToken ? { credentials: { accessToken } } : {}),
      }) as AsyncGenerator<unknown>;
    },
    [useCuratedOnly, hasQuery, query, accessToken, sortBy, sortDirection],
  );

  const search = useHubPaginatedSearch(createIter, mapDataset, { enabled });

  const results = useMemo(() => {
    if (!enabled) return [];
    const hideOcr = modelType !== "vision";
    const baseResults = hideOcr
      ? search.results.filter((ds) => !isOcrOrVisionTextDataset(ds))
      : search.results;

    if (!hasQuery && modelType) {
      const curatedIds = CURATED_EMPTY_QUERY_DATASET_IDS[modelType] ?? [];
      return curatedIds.map(toCuratedDatasetResult);
    }

    if (!modelType) return baseResults;

    const boosted: HfDatasetResult[] = [];
    const neutral: HfDatasetResult[] = [];

    for (const ds of baseResults) {
      const relevance = rankDatasetRelevance(ds, modelType);
      if (relevance === "boosted") boosted.push(ds);
      else if (relevance !== "incompatible") neutral.push(ds);
    }

    return [...boosted, ...neutral];
  }, [enabled, search.results, modelType, query]);

  return { ...search, results };
}
