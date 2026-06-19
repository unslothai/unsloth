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

// Embedding / sentence-transformer models ship with onnx/openvino as additional
// export formats — they should not be excluded by the tag check above.
const EMBEDDING_TAGS = new Set([
  "sentence-transformers",
  "feature-extraction",
]);

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
      tags?: string[];
    };
    const isEmbedding = m.tags?.some((t) => EMBEDDING_TAGS.has(t));
    if (!isEmbedding && m.tags?.some((t) => excludedTags.has(t))) {
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
    };
  };
}

/** Number of unsloth results to pull up-front before yielding general results. */
const UNSLOTH_PREFETCH = 20;
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
): AsyncGenerator<unknown> {
  const common = {
    additionalFields: ["safetensors", "tags"] as ("safetensors" | "tags")[],
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

  // Start pinned model lookup immediately so it can run in parallel with
  // the Phase 1 unsloth iteration instead of blocking Phase 2.
  const pinnedPromise = pinnedId
    ? cachedModelInfo({
        name: pinnedId,
        additionalFields: ["safetensors", "tags"],
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
): AsyncGenerator<unknown> {
  const common = {
    additionalFields: ["safetensors", "tags"] as ("safetensors" | "tags")[],
    fetch: withPopularitySort,
    ...(accessToken ? { credentials: { accessToken } } : {}),
  };

  // Phase 1: fetch priority models in parallel via modelInfo
  const seen = new Set<string>();
  const settled = await Promise.allSettled(
    priorityIds.map((id) =>
      cachedModelInfo({
        name: id,
        additionalFields: ["safetensors", "tags"],
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

export function useHfModelSearch(
  query: string,
  options?: {
    task?: PipelineType;
    accessToken?: string;
    excludeGguf?: boolean;
    priorityIds?: readonly string[];
  },
) {
  const { task, accessToken, excludeGguf = false, priorityIds } = options ?? {};

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
      if (!trimmed) {
        // No query: show priority models first (with full metadata), then general unsloth listing
        if (priorityIds && priorityIds.length > 0) {
          return priorityThenListingIterator(priorityIds, task, accessToken) as AsyncGenerator<unknown>;
        }
        return listModels({
          search: { owner: "unsloth", ...(task ? { task } : {}) },
          additionalFields: ["safetensors", "tags"],
          fetch: withPopularitySort,
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
      return mergedModelIterator(searchQuery, undefined, accessToken, pinnedId) as AsyncGenerator<unknown>;
    },
    [trimmed, searchQuery, pinnedId, task, accessToken, priorityIds],
  );

  const deviceType = usePlatformStore((s) => s.deviceType);
  const excludedTags = deviceType === "mac" ? EXCLUDED_TAGS_MLX : EXCLUDED_TAGS_GPU;
  const mapModel = useMemo(() => makeMapModel(excludeGguf, excludedTags), [excludeGguf, excludedTags]);
  const search = useHfPaginatedSearch(createIter, mapModel);

  // Secondary sort guarantee: unsloth models always float to the top.
  // Skip when the user searched for a specific non-unsloth publisher
  // (e.g. "openai/gpt-oss-20b") -- the iterator already handles the
  // pinned ordering in that case.
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
