// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PipelineType } from "@huggingface/hub";
import { listModels } from "@huggingface/hub";
import {
  ALL_FIELDS,
  type CachedResult,
  cachedModelInfo,
  primeCacheFromListing,
} from "@/lib/hf-cache";
import { useCallback, useMemo } from "react";
import { useHfPaginatedSearch } from "./use-hf-paginated-search";
import { usePlatformStore } from "@/config/env";
import {
  EMBEDDING_TAGS,
  estimateSizeFromDtypes,
} from "@/features/models/lib/hf-model-meta";

export type HfSortKey =
  | "trendingScore"
  | "downloads"
  | "likes"
  | "lastModified"
  | "createdAt";

export type HfSortDirection = "desc" | "asc";

export interface HfModelResult {
  id: string;
  downloads: number;
  likes: number;
  totalParams?: number;
  estimatedSizeBytes?: number;
  isGguf: boolean;
  tags?: string[];
  pipelineTag?: string;
  updatedAt?: string;
  libraryName?: string;
  quantMethod?: string;
}

/** Tags to exclude on GPU (CUDA/ROCm) — MLX models won't load on GPU. */
const EXCLUDED_TAGS_GPU = new Set([
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

/** Tags to exclude on MLX (Mac) — GPU-only quant formats won't load on MLX. */
const EXCLUDED_TAGS_MLX = new Set([
  "gptq",
  "awq",
  "exl2",
  "onnx",
  "openvino",
  "coreml",
  "tflite",
  "ctranslate2",
]);

const UNSUPPORTED_PIPELINE_TAGS: ReadonlySet<string> = new Set([
  "text-to-image",
  "image-to-image",
  "image-text-to-image",
  "text-to-video",
  "video-to-video",
  "image-to-video",
  "video-text-to-text",
  "video-classification",
  "unconditional-image-generation",
  "text-to-3d",
  "image-to-3d",
  "image-segmentation",
  "object-detection",
  "depth-estimation",
  "mask-generation",
  "zero-shot-image-classification",
  "zero-shot-object-detection",
  "image-classification",
  "keypoint-detection",
  "image-feature-extraction",
  "robotics",
  "reinforcement-learning",
  "graph-ml",
  "tabular-classification",
  "tabular-regression",
  "time-series-forecasting",
]);

const UNSUPPORTED_LIBRARY_TAGS: ReadonlySet<string> = new Set([
  "diffusers",
  "stable-diffusion",
  "stable-diffusion-xl",
  "flux",
  "controlnet",
  "lora-diffusers",
]);

const FORMAT_TAG_LABEL: Record<string, string> = {
  gptq: "GPTQ quantization",
  awq: "AWQ quantization",
  exl2: "EXL2 quantization",
  mlx: "MLX-format weights",
  onnx: "ONNX-format weights",
  openvino: "OpenVINO-format weights",
  coreml: "Core ML-format weights",
  tflite: "TensorFlow Lite-format weights",
  ctranslate2: "CTranslate2-format weights",
};

const FORMAT_ALIAS_TAGS: Record<string, string> = {
  "auto-gptq": "gptq",
};

const FORMAT_NAME_PATTERNS: ReadonlyArray<{ key: string; pattern: RegExp }> = [
  { key: "awq", pattern: /(?:^|[-_./])awq(?:\d+(?:bit)?)?(?:$|[-_./])/i },
  { key: "gptq", pattern: /(?:^|[-_./])gptq(?:\d+(?:bit)?)?(?:$|[-_./])/i },
  { key: "exl2", pattern: /(?:^|[-_./])exl2(?:$|[-_./])/i },
  { key: "mlx", pattern: /(?:^|[-_./])mlx(?:$|[-_./])/i },
  { key: "onnx", pattern: /(?:^|[-_./])onnx(?:$|[-_./])/i },
  { key: "openvino", pattern: /(?:^|[-_./])openvino(?:$|[-_./])/i },
  { key: "coreml", pattern: /(?:^|[-_./])coreml(?:$|[-_./])/i },
  { key: "tflite", pattern: /(?:^|[-_./])tflite(?:$|[-_./])/i },
  { key: "ctranslate2", pattern: /(?:^|[-_./])ctranslate2(?:$|[-_./])/i },
];

function detectFormatKey(
  modelId: string | null | undefined,
  lowerTags: ReadonlySet<string>,
): string | null {
  for (const tag of lowerTags) {
    if (FORMAT_TAG_LABEL[tag]) return tag;
    const alias = FORMAT_ALIAS_TAGS[tag];
    if (alias) return alias;
  }
  if (modelId) {
    for (const { key, pattern } of FORMAT_NAME_PATTERNS) {
      if (pattern.test(modelId)) return key;
    }
  }
  return null;
}

/**
 * `config.quantization_config.quant_method` values Unsloth can load natively.
 * Anything matching one of these stays unflagged regardless of the model name
 * or tags — this is the authoritative signal because it comes from the model's
 * own config.json. See
 * https://huggingface.co/docs/transformers/quantization/overview.
 */
const SUPPORTED_QUANT_METHODS: ReadonlySet<string> = new Set([
  "bitsandbytes",
  "bnb",
  "bnb_4bit",
  "bnb_8bit",
]);

/**
 * `quant_method` values that Unsloth's runtime cannot load today. Maps to the
 * label rendered to the user. Methods not in this map and not in
 * `SUPPORTED_QUANT_METHODS` are treated as "unknown" and *not* flagged, so a
 * brand-new method (or a publisher typo) never produces a false negative
 * support claim.
 */
const UNSUPPORTED_QUANT_METHODS: Record<string, string> = {
  awq: "AWQ quantization",
  gptq: "GPTQ quantization",
  exl2: "EXL2 quantization",
  "compressed-tensors": "compressed-tensors quantization",
  aqlm: "AQLM quantization",
  eetq: "EETQ quantization",
  hqq: "HQQ quantization",
  fbgemm_fp8: "FBGEMM FP8 quantization",
  finegrained_fp8: "fine-grained FP8 quantization",
  quark: "Quark quantization",
  vptq: "VPTQ quantization",
  spqr: "SpQR quantization",
  higgs: "HIGGS quantization",
  sinq: "SINQ quantization",
  torchao: "torchao quantization",
  quanto: "optimum-quanto quantization",
  auto_round: "AutoRound quantization",
  autoround: "AutoRound quantization",
  metal: "Metal-kernel quantization",
  fouroversix: "Four Over Six quantization",
  fp_quant: "FP-Quant quantization",
};

function normalizeQuantMethod(value: string | null | undefined): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.toLowerCase().trim();
  return trimmed.length > 0 ? trimmed : null;
}

export type UnslothSupportStatus = "supported" | "unsupported";

export interface UnslothSupport {
  status: UnslothSupportStatus;
  reason: string | null;
}

/**
 * Classify a Hugging Face model as supported by Unsloth on the active device.
 * The Hub uses this to render a non-blocking notice; the training picker
 * filters out unsupported models via separate task/format filters.
 */
export function classifyUnslothSupport({
  modelId,
  pipelineTag,
  tags,
  libraryName,
  deviceType,
  quantMethod,
}: {
  modelId?: string | null;
  pipelineTag?: string | null;
  tags?: readonly string[] | null;
  libraryName?: string | null;
  deviceType?: string | null;
  quantMethod?: string | null;
}): UnslothSupport {
  const pipeline = pipelineTag?.toLowerCase().trim() || null;
  const lowerTags = new Set(
    (tags ?? []).map((tag) => tag.toLowerCase().trim()).filter(Boolean),
  );
  const library = libraryName?.toLowerCase().trim() || null;
  const formatTags =
    deviceType?.toLowerCase() === "mac"
      ? EXCLUDED_TAGS_MLX
      : EXCLUDED_TAGS_GPU;
  const normalizedQuant = normalizeQuantMethod(quantMethod);

  if (normalizedQuant) {
    if (SUPPORTED_QUANT_METHODS.has(normalizedQuant)) {
      return { status: "supported", reason: null };
    }
    if (Object.hasOwn(UNSUPPORTED_QUANT_METHODS, normalizedQuant)) {
      return {
        status: "unsupported",
        reason: `Detected ${UNSUPPORTED_QUANT_METHODS[normalizedQuant]}.`,
      };
    }
  }

  if (pipeline && UNSUPPORTED_PIPELINE_TAGS.has(pipeline)) {
    return {
      status: "unsupported",
      reason: `Pipeline task: ${pipeline}.`,
    };
  }
  for (const tag of lowerTags) {
    if (UNSUPPORTED_LIBRARY_TAGS.has(tag)) {
      return {
        status: "unsupported",
        reason: `Library: ${tag}.`,
      };
    }
  }
  if (library && UNSUPPORTED_LIBRARY_TAGS.has(library)) {
    return {
      status: "unsupported",
      reason: `Library: ${library}.`,
    };
  }
  const formatKey = detectFormatKey(modelId, lowerTags);
  if (formatKey && formatTags.has(formatKey)) {
    const label = FORMAT_TAG_LABEL[formatKey] ?? `${formatKey.toUpperCase()} weights`;
    return {
      status: "unsupported",
      reason: `Detected ${label}.`,
    };
  }
  return { status: "supported", reason: null };
}

// HF rejects direction=1 (asc) for trendingScore — only descending is supported.
const DESC_ONLY_SORTS = new Set<HfSortKey>(["trendingScore"]);

function makeSortFetch(
  sortBy: HfSortKey | undefined,
  direction: HfSortDirection,
): typeof fetch {
  return (input, init) => {
    const rawUrl =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
    const url = new URL(rawUrl);

    if (sortBy && !url.searchParams.has("sort")) {
      url.searchParams.set("sort", sortBy);
    }
    const effectiveSort = (url.searchParams.get("sort") ?? sortBy) as
      | HfSortKey
      | undefined;
    const effectiveDir =
      effectiveSort && DESC_ONLY_SORTS.has(effectiveSort) ? "desc" : direction;
    url.searchParams.set("direction", effectiveDir === "asc" ? "1" : "-1");

    return fetch(url, init);
  };
}

const withPopularitySort = makeSortFetch("downloads", "desc");

function makeMapModel(
  excludeGguf: boolean,
  excludedTags: ReadonlySet<string>,
  keepUnsupportedTags: boolean,
  idSuffix: string,
) {
  const suffixLower = idSuffix.toLowerCase();
  return (raw: unknown): HfModelResult | null => {
    const m = raw as {
      name: string;
      downloads: number;
      likes: number;
      task?: string;
      pipeline_tag?: string;
      library_name?: string;
      updatedAt?: Date | string;
      safetensors?: { total: number; parameters?: Record<string, number> };
      gguf?: { total?: number; architecture?: string };
      tags?: string[];
      config?: { quantization_config?: { quant_method?: string } };
    };
    if (suffixLower && !m.name?.toLowerCase().endsWith(suffixLower)) {
      return null;
    }
    const isEmbedding = m.tags?.some((t) => EMBEDDING_TAGS.has(t));
    if (
      !keepUnsupportedTags &&
      !isEmbedding &&
      m.tags?.some((t) => excludedTags.has(t))
    ) {
      return null;
    }
    const isGguf =
      Boolean(m.tags?.some((tag) => tag.toLowerCase() === "gguf")) ||
      /-GGUF(?:$|-)/i.test(m.name);
    if (excludeGguf && isGguf) {
      return null;
    }
    const updatedAtIso =
      m.updatedAt instanceof Date
        ? m.updatedAt.toISOString()
        : typeof m.updatedAt === "string"
          ? m.updatedAt
          : undefined;
    return {
      id: m.name,
      downloads: m.downloads,
      likes: m.likes,
      totalParams: m.safetensors?.total ?? m.gguf?.total,
      estimatedSizeBytes: estimateSizeFromDtypes(m.safetensors?.parameters),
      isGguf,
      tags: m.tags,
      pipelineTag: m.task ?? m.pipeline_tag,
      updatedAt: updatedAtIso,
      libraryName: m.library_name,
      quantMethod: m.config?.quantization_config?.quant_method,
    };
  };
}

/** Number of unsloth results to pull up-front before yielding general results. */
const UNSLOTH_PREFETCH = 20;
/** When the user typed a query, only float a few unsloth results before
 *  yielding the general (relevance-ranked) listing. */
const UNSLOTH_QUERY_PREFETCH = 3;
/** When the user searched for a specific publisher, show fewer unsloth results
 *  before the pinned (original publisher) model. */
const UNSLOTH_PINNED_PREFETCH = 4;
/** Matches a valid "owner/repo" identifier (exactly two non-empty segments). */
const PUBLISHER_RE = /^([^/\s]+)\/([^/\s]+)$/;

/**
 * Prime the hf-cache from a listModels result. For public (non-gated,
 * non-private) models, also prime the anonymous slot so the VRAM hook
 * gets cache hits without re-fetching. Gated/private models are only
 * cached under the caller's token to avoid auth leakage.
 */
function primeFromListing(
  name: string,
  accessToken: string | undefined,
  model: unknown,
): void {
  const data = model as CachedResult;
  primeCacheFromListing(name, accessToken, data);
  if (accessToken && !data.private && !data.gated) {
    primeCacheFromListing(name, undefined, data);
  }
}

/**
 * Creates a merged async generator that yields unsloth-owned models first,
 * then general results (with deduplication).
 */
async function* mergedModelIterator(
  query: string,
  task?: PipelineType,
  accessToken?: string,
  pinnedId?: string,
  sortBy: HfSortKey = "downloads",
  direction: HfSortDirection = "desc",
): AsyncGenerator<unknown> {
  const sortFetch = makeSortFetch(sortBy, direction);
  const common = {
    additionalFields: ALL_FIELDS,
    fetch: sortFetch,
    ...(accessToken ? { credentials: { accessToken } } : {}),
  };

  // Fire both iterators immediately (parallel network requests on first pull)
  const unslothIter = listModels({
    search: { query, owner: "unsloth", ...(task ? { task } : {}) },
    ...common,
  });
  const generalIter = listModels({
    search: { query, ...(task ? { task } : {}) },
    ...common,
  });

  // Start pinned model lookup immediately so it can run in parallel with
  // the Phase 1 unsloth iteration instead of blocking Phase 2.
  const pinnedPromise = pinnedId
    ? cachedModelInfo({
        name: pinnedId,
        additionalFields: ALL_FIELDS,
        ...(accessToken ? { credentials: { accessToken } } : {}),
      }).catch(() => null)
    : null;

  const limit = pinnedId
    ? UNSLOTH_PINNED_PREFETCH
    : query.trim()
      ? UNSLOTH_QUERY_PREFETCH
      : UNSLOTH_PREFETCH;

  // Phase 1: pull & yield unsloth models first
  const seen = new Set<string>();
  let count = 0;
  for await (const model of unslothIter) {
    const m = model as { name?: string };
    if (m.name) {
      seen.add(m.name);
      primeFromListing(m.name, accessToken, model);
    }
    yield model;
    count++;
    if (count >= limit) break;
  }

  // Phase 1b: yield the pinned (original publisher) model before general results
  if (pinnedId && !seen.has(pinnedId) && pinnedPromise) {
    const pinned = await pinnedPromise;
    if (pinned) {
      // Record both the raw input and the canonical name returned by HF
      // so phase 2 deduplication works even when casing differs
      // (e.g. user typed "OpenAI/gpt-oss-20b", HF returns "openai/gpt-oss-20b").
      seen.add(pinnedId);
      const canonicalName = (pinned as { name?: string }).name;
      if (canonicalName && canonicalName !== pinnedId) {
        seen.add(canonicalName);
      }
      yield pinned;
    }
  }

  // Phase 2: yield general results, skipping already-seen models
  for await (const model of generalIter) {
    const m = model as { name?: string };
    if (m.name && seen.has(m.name)) continue;
    if (m.name) {
      primeFromListing(m.name, accessToken, model);
    }
    yield model;
  }
}

/**
 * Creates an async generator that yields priority models (fetched individually
 * via modelInfo for full metadata), then the general unsloth listing.
 */
async function* priorityThenListingIterator(
  priorityIds: readonly string[],
  task?: PipelineType,
  accessToken?: string,
  sortBy: HfSortKey = "downloads",
  direction: HfSortDirection = "desc",
): AsyncGenerator<unknown> {
  const common = {
    additionalFields: ALL_FIELDS,
    fetch: makeSortFetch(sortBy, direction),
    ...(accessToken ? { credentials: { accessToken } } : {}),
  };

  // Phase 1: fetch priority models in parallel via modelInfo
  const seen = new Set<string>();
  const settled = await Promise.allSettled(
    priorityIds.map((id) =>
      cachedModelInfo({
        name: id,
        additionalFields: ALL_FIELDS,
        ...(accessToken ? { credentials: { accessToken } } : {}),
      }),
    ),
  );
  for (const result of settled) {
    if (result.status === "fulfilled") {
      const m = result.value as { name?: string; pipeline_tag?: string };
      // Skip models that don't match the selected task filter
      if (task && m.pipeline_tag && m.pipeline_tag !== task) continue;
      if (m.name) seen.add(m.name);
      yield result.value;
    }
  }

  // Phase 2: yield general unsloth listing, skipping already-seen
  const generalIter = listModels({
    search: { owner: "unsloth", ...(task ? { task } : {}) },
    ...common,
  });
  for await (const model of generalIter) {
    const m = model as { name?: string };
    if (m.name && seen.has(m.name)) continue;
    if (m.name) {
      primeFromListing(m.name, accessToken, model);
    }
    yield model;
  }
}

export interface HfModelSearchChannel {
  owner?: string;
  tags?: readonly string[];
  /** Free-text query injected into the HF listModels search.query field. */
  query?: string;
  /** Strict client-side filter: drop results whose id doesn't end with this. */
  idSuffix?: string;
}

export function useHfModelSearch(
  query: string,
  options?: {
    task?: PipelineType;
    accessToken?: string;
    excludeGguf?: boolean;
    priorityIds?: readonly string[];
    sortBy?: HfSortKey;
    sortDirection?: HfSortDirection;
    pinUnslothFirst?: boolean;
    enabled?: boolean;
    keepUnsupportedTags?: boolean;
    channel?: HfModelSearchChannel | null;
  },
) {
  const {
    task,
    accessToken,
    excludeGguf = false,
    priorityIds,
    sortBy = "downloads",
    sortDirection = "desc",
    pinUnslothFirst = true,
    enabled = true,
    keepUnsupportedTags = false,
    channel = null,
  } = options ?? {};

  const channelOwner = channel?.owner ?? null;
  const channelTagsKey = channel?.tags ? channel.tags.join("|") : "";
  const channelQuery = channel?.query ?? "";
  const channelIdSuffix = channel?.idSuffix ?? "";

  // Parse publisher detection once and share between the iterator factory
  // and the secondary sort gate (avoids duplicating the regex + logic).
  const { isPublisherQuery, searchQuery, pinnedId, trimmed } = useMemo(() => {
    const t = query.trim();
    const m = PUBLISHER_RE.exec(t);
    const is = !!m && m[1].toLowerCase() !== "unsloth";
    return {
      isPublisherQuery: is,
      searchQuery: is ? m![2] : t,
      pinnedId: is ? t : undefined,
      trimmed: t,
    };
  }, [query]);

  const createIter = useCallback(
    () => {
      // Channel scoping bypasses the unsloth-merge iterator: listings get a
      // hard owner/tag filter so the sidebar shows just that curated slice.
      // The user's text query (if any) is forwarded as a free-text filter
      // within the channel.
      if (channelOwner || channelTagsKey || channelQuery) {
        const channelTags = channelTagsKey ? channelTagsKey.split("|") : undefined;
        // User text query takes precedence over channel.query (so typing inside
        // a channel still refines the slice). Without user input, the channel's
        // own query (e.g. "bnb-4bit") narrows the listing server-side.
        const queryString = trimmed || channelQuery || undefined;
        return listModels({
          search: {
            ...(queryString ? { query: queryString } : {}),
            ...(channelOwner ? { owner: channelOwner } : {}),
            ...(channelTags ? { tags: channelTags } : {}),
          },
          additionalFields: ALL_FIELDS,
          fetch: makeSortFetch(sortBy, sortDirection),
          sort: sortBy,
          ...(accessToken ? { credentials: { accessToken } } : {}),
        }) as AsyncGenerator<unknown>;
      }
      if (!trimmed) {
        // No query: show priority models first (with full metadata), then general unsloth listing
        if (priorityIds && priorityIds.length > 0) {
          return priorityThenListingIterator(
            priorityIds,
            task,
            accessToken,
            sortBy,
            sortDirection,
          ) as AsyncGenerator<unknown>;
        }
        return listModels({
          search: { ...(task ? { task } : {}) },
          additionalFields: ALL_FIELDS,
          fetch: makeSortFetch(sortBy, sortDirection),
          sort: sortBy,
          ...(accessToken ? { credentials: { accessToken } } : {}),
        }) as AsyncGenerator<unknown>;
      }
      // Typed query: disable task filter so explicitly searched models still
      // appear even if HF task metadata is wrong/missing.
      // If the query is a valid "owner/repo" identifier (exactly two non-empty,
      // slash-free, space-free segments), strip the org prefix so unsloth
      // variants surface, then pin the original publisher model after a small
      // batch of unsloth results.  Queries for unsloth-owned models are left
      // as-is so they get the full 20-result prefetch and secondary sort.
      return mergedModelIterator(
        searchQuery,
        undefined,
        accessToken,
        pinnedId,
        sortBy,
        sortDirection,
      ) as AsyncGenerator<unknown>;
    },
    [
      trimmed,
      searchQuery,
      pinnedId,
      task,
      accessToken,
      priorityIds,
      sortBy,
      sortDirection,
      channelOwner,
      channelTagsKey,
      channelQuery,
    ],
  );

  const deviceType = usePlatformStore((s) => s.deviceType);
  const excludedTags =
    deviceType === "mac" ? EXCLUDED_TAGS_MLX : EXCLUDED_TAGS_GPU;
  const mapModel = useMemo(
    () =>
      makeMapModel(
        excludeGguf,
        excludedTags,
        keepUnsupportedTags,
        channelIdSuffix,
      ),
    [excludeGguf, excludedTags, keepUnsupportedTags, channelIdSuffix],
  );
  const search = useHfPaginatedSearch(createIter, mapModel, { enabled });

  // Secondary sort: only when there's no user query. With a query, the merged
  // iterator already floats a small number of unsloth results to the top —
  // re-sorting all loaded results would bury the user's actual search matches.
  // Channel scoping has its own ordering, so the re-sort is also disabled there.
  const channelActive = Boolean(channelOwner || channelTagsKey);
  const results = useMemo(
    () =>
      !pinUnslothFirst || isPublisherQuery || trimmed || channelActive
        ? search.results
        : [...search.results].sort((a, b) => {
            const aFirst = a.id.startsWith("unsloth/") ? 0 : 1;
            const bFirst = b.id.startsWith("unsloth/") ? 0 : 1;
            return aFirst - bFirst;
          }),
    [search.results, isPublisherQuery, trimmed, channelActive, pinUnslothFirst],
  );

  return { ...search, results };
}
