// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PipelineType } from "@huggingface/hub";
import { listModels } from "@huggingface/hub";
import { type CachedResult, cachedModelInfo, primeCacheFromListing } from "@/lib/hf-cache";
import { useCallback, useMemo } from "react";
import { useHfPaginatedSearch } from "./use-hf-paginated-search";
import { usePlatformStore } from "@/config/env";

export interface HfModelResult {
  id: string;
  downloads: number;
  likes: number;
  totalParams?: number;
  estimatedSizeBytes?: number;
  isGguf: boolean;
  tags?: string[];
  pipelineTag?: string;
  /** Param count from GGUF metadata, for repos with no safetensors weights. */
  ggufParams?: number;
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

// Embedding / sentence-transformer models ship onnx/openvino as extra
// export formats, so the tag check above must not exclude them.
const EMBEDDING_TAGS = new Set([
  "sentence-transformers",
  "feature-extraction",
]);

// Image/video/3D generation tasks. These are diffusion models that cannot run
// in chat, so we drop them from the listing (matched by pipeline tag or tag).
const EXCLUDED_TASKS = new Set([
  "text-to-image",
  "image-to-image",
  "text-to-video",
  "image-to-video",
  "unconditional-image-generation",
  "text-to-3d",
  "image-to-3d",
]);

// Name fallback for well-known diffusion/video families with no task tag.
const DIFFUSION_NAME_RE =
  /(?:^|[-_/. ])(?:ltx(?:v|-video)?|flux|stable-diffusion|sdxl|sd3|wan2|hunyuan-?video|cogvideo|mochi|animatediff|qwen-image)(?=$|[-_/. ])/i;

function withPopularitySort(
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
    url.searchParams.set("sort", "downloads");
  }
  if (!url.searchParams.has("direction")) {
    url.searchParams.set("direction", "-1");
  }

  return fetch(url, init);
}

export type HfModelSort =
  | "trendingScore"
  | "downloads"
  | "likes"
  | "lastModified"
  | "createdAt";

/** Like withPopularitySort but forces a specific sort key (descending). */
function makeSortFetch(sort: HfModelSort) {
  return (
    input: Parameters<typeof fetch>[0],
    init?: Parameters<typeof fetch>[1],
  ): ReturnType<typeof fetch> => {
    const rawUrl =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
    const url = new URL(rawUrl);
    url.searchParams.set("sort", sort);
    if (!url.searchParams.has("direction")) {
      url.searchParams.set("direction", "-1");
    }
    return fetch(url, init);
  };
}

/** Bytes per parameter for each dtype. */
const DTYPE_BYTES: Record<string, number> = {
  F64: 8, F32: 4, F16: 2, BF16: 2,
  I64: 8, I32: 4, I16: 2, I8: 1, U8: 1,
  // Quantized types (4-bit)
  NF4: 0.5, FP4: 0.5, INT4: 0.5, GPTQ: 0.5,
};

function estimateSizeFromDtypes(
  params: Record<string, number> | undefined,
): number | undefined {
  if (!params) return undefined;
  let total = 0;
  for (const [dtype, count] of Object.entries(params)) {
    const bpp = DTYPE_BYTES[dtype.toUpperCase()] ?? 2; // default BF16
    total += count * bpp;
  }
  return total > 0 ? total : undefined;
}

function makeMapModel(excludeGguf: boolean, excludedTags: Set<string>) {
  return (raw: unknown): HfModelResult | null => {
    const m = raw as {
      name: string;
      downloads: number;
      likes: number;
      safetensors?: { total: number; parameters?: Record<string, number> };
      gguf?: { total?: number };
      tags?: string[];
      pipeline_tag?: string;
    };
    const isEmbedding = m.tags?.some((t) => EMBEDDING_TAGS.has(t));
    if (!isEmbedding && m.tags?.some((t) => excludedTags.has(t))) {
      return null;
    }
    // Drop image/video diffusion models; they cannot run in chat.
    const isDiffusion =
      (m.pipeline_tag && EXCLUDED_TASKS.has(m.pipeline_tag.toLowerCase())) ||
      m.tags?.some((t) => EXCLUDED_TASKS.has(t.toLowerCase())) ||
      DIFFUSION_NAME_RE.test(m.name);
    if (isDiffusion) {
      return null;
    }
    const isGguf =
      Boolean(m.tags?.some((tag) => tag.toLowerCase() === "gguf")) ||
      /-GGUF(?:$|-)/i.test(m.name);
    if (excludeGguf && isGguf) {
      return null;
    }
    return {
      id: m.name,
      downloads: m.downloads,
      likes: m.likes,
      totalParams: m.safetensors?.total,
      estimatedSizeBytes: estimateSizeFromDtypes(m.safetensors?.parameters),
      isGguf,
      tags: m.tags,
      pipelineTag: m.pipeline_tag,
      ggufParams: m.gguf?.total,
    };
  };
}

// HF expand fields to request. "gguf" returns a repo's GGUF param count (so
// repos with no safetensors weights can still be sized); it works at runtime
// but is missing from the hub client's type union, so we widen it here.
const MODEL_FIELDS = ["safetensors", "tags", "gguf"] as unknown as ("safetensors" | "tags")[];

/** Number of unsloth results to pull up-front before yielding general results. */
const UNSLOTH_PREFETCH = 20;
/** Fewer unsloth results before the pinned model when a publisher is searched. */
const UNSLOTH_PINNED_PREFETCH = 4;
/** Matches a valid "owner/repo" identifier (exactly two non-empty segments). */
const PUBLISHER_RE = /^([^/\s]+)\/([^/\s]+)$/;

/**
 * Prime the hf-cache from a listModels result. For public models, also
 * prime the anonymous slot so the VRAM hook gets cache hits without
 * re-fetching. Gated/private models are cached only under the caller's
 * token to avoid auth leakage.
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
): AsyncGenerator<unknown> {
  const common = {
    additionalFields: MODEL_FIELDS,
    fetch: withPopularitySort,
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

  // Start the pinned lookup now so it runs in parallel with Phase 1
  // instead of blocking Phase 2.
  const pinnedPromise = pinnedId
    ? cachedModelInfo({
        name: pinnedId,
        additionalFields: MODEL_FIELDS,
        ...(accessToken ? { credentials: { accessToken } } : {}),
      }).catch(() => null)
    : null;

  const limit = pinnedId ? UNSLOTH_PINNED_PREFETCH : UNSLOTH_PREFETCH;

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
      // Record both the raw input and HF's canonical name so phase 2 dedupe
      // works across casing (e.g. "OpenAI/..." vs HF's "openai/...").
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
): AsyncGenerator<unknown> {
  const common = {
    additionalFields: MODEL_FIELDS,
    fetch: withPopularitySort,
    ...(accessToken ? { credentials: { accessToken } } : {}),
  };

  // Phase 1: fetch priority models in parallel via modelInfo
  const seen = new Set<string>();
  const settled = await Promise.allSettled(
    priorityIds.map((id) =>
      cachedModelInfo({
        name: id,
        additionalFields: MODEL_FIELDS,
        ...(accessToken ? { credentials: { accessToken } } : {}),
      }),
    ),
  );
  for (const result of settled) {
    if (result.status === "fulfilled") {
      const m = result.value as { name?: string; pipeline_tag?: string };
      // Skip models that don't match the task filter.
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

export function useHfModelSearch(
  query: string,
  options?: {
    task?: PipelineType;
    accessToken?: string;
    excludeGguf?: boolean;
    priorityIds?: readonly string[];
    sort?: HfModelSort;
  },
) {
  const {
    task,
    accessToken,
    excludeGguf = false,
    priorityIds,
    sort = "downloads",
  } = options ?? {};

  // Detect the publisher once, shared by the iterator factory and the
  // secondary sort gate (avoids duplicating the regex + logic).
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
      if (!trimmed) {
        // No query: priority models first (full metadata), then general unsloth.
        if (priorityIds && priorityIds.length > 0) {
          return priorityThenListingIterator(priorityIds, task, accessToken) as AsyncGenerator<unknown>;
        }
        return listModels({
          search: { owner: "unsloth", ...(task ? { task } : {}) },
          additionalFields: MODEL_FIELDS,
          fetch: makeSortFetch(sort),
          sort,
          ...(accessToken ? { credentials: { accessToken } } : {}),
        }) as AsyncGenerator<unknown>;
      }
      // Typed query: disable the task filter so explicitly searched models
      // appear even with wrong/missing HF task metadata. For a valid
      // "owner/repo" query, strip the org prefix so unsloth variants
      // surface, then pin the original publisher model after a small batch.
      // unsloth-owned queries are left as-is for the full prefetch + sort.
      return mergedModelIterator(searchQuery, undefined, accessToken, pinnedId) as AsyncGenerator<unknown>;
    },
    [trimmed, searchQuery, pinnedId, task, accessToken, priorityIds, sort],
  );

  const deviceType = usePlatformStore((s) => s.deviceType);
  const excludedTags = deviceType === "mac" ? EXCLUDED_TAGS_MLX : EXCLUDED_TAGS_GPU;
  const mapModel = useMemo(() => makeMapModel(excludeGguf, excludedTags), [excludeGguf, excludedTags]);
  const search = useHfPaginatedSearch(createIter, mapModel);

  // Secondary sort: unsloth models always float to the top. Skip for a
  // specific non-unsloth publisher query (e.g. "openai/gpt-oss-20b") --
  // the iterator already handles pinned ordering there.
  const results = useMemo(
    () =>
      isPublisherQuery
        ? search.results
        : [...search.results].sort((a, b) => {
            const aFirst = a.id.startsWith("unsloth/") ? 0 : 1;
            const bFirst = b.id.startsWith("unsloth/") ? 0 : 1;
            return aFirst - bFirst;
          }),
    [search.results, isPublisherQuery],
  );

  return { ...search, results };
}
