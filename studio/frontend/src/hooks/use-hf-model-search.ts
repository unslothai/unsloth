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
import { fetchWithTimeout } from "@/lib/network";
import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { useHfPaginatedSearch } from "./use-hf-paginated-search";
import { usePlatformStore } from "@/config/env";
import {
  EMBEDDING_TAGS,
  estimateSizeFromDtypes,
} from "@/lib/hf-model-meta";
import { isGgufLike } from "@/lib/model-identifiers";
import {
  classifyUnslothSupport,
  excludedFormatTagsForDevice,
  type UnslothSupport,
  type UnslothSupportStatus,
} from "@/lib/unsloth-support";

export { classifyUnslothSupport };
export type { UnslothSupport, UnslothSupportStatus };

export type HfSortKey =
  | "trendingScore"
  | "downloads"
  | "likes"
  | "lastModified"
  | "createdAt";

export type HfSortDirection = "desc" | "asc";
export type HfTaskFilter = PipelineType | readonly PipelineType[] | undefined;

function normalizeTaskFilter(task: HfTaskFilter): readonly PipelineType[] {
  if (!task) return [];
  return typeof task === "string" ? [task] : task;
}

function taskMatches(
  pipelineTag: string | undefined,
  tasks: readonly PipelineType[],
): boolean {
  return (
    tasks.length === 0 ||
    !pipelineTag ||
    tasks.includes(pipelineTag as PipelineType)
  );
}

async function* mergeTaskIterators(
  tasks: readonly PipelineType[],
  createIter: (task?: PipelineType) => AsyncGenerator<unknown>,
): AsyncGenerator<unknown> {
  const seen = new Set<string>();
  const taskList = tasks.length > 0 ? tasks : [undefined];
  for (const task of taskList) {
    for await (const model of createIter(task)) {
      const name = (model as { name?: string }).name;
      if (name) {
        const key = name.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
      }
      yield model;
    }
  }
}

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

// HF rejects direction=1 (asc) for trendingScore — only descending is supported.
const DESC_ONLY_SORTS = new Set<HfSortKey>(["trendingScore"]);
const HF_SEARCH_TIMEOUT_MS = 15_000;

function makeHfFetch(signal?: AbortSignal): typeof fetch {
  return (input, init) =>
    fetchWithTimeout(
      input,
      signal ? { ...init, signal } : init,
      HF_SEARCH_TIMEOUT_MS,
    );
}

function makeSortFetch(
  sortBy: HfSortKey | undefined,
  direction: HfSortDirection,
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

    if (sortBy && !url.searchParams.has("sort")) {
      url.searchParams.set("sort", sortBy);
    }
    const effectiveSort = (url.searchParams.get("sort") ?? sortBy) as
      | HfSortKey
      | undefined;
    const effectiveDir =
      effectiveSort && DESC_ONLY_SORTS.has(effectiveSort) ? "desc" : direction;
    url.searchParams.set("direction", effectiveDir === "asc" ? "1" : "-1");

    return fetchWithTimeout(
      url,
      signal ? { ...init, signal } : init,
      HF_SEARCH_TIMEOUT_MS,
    );
  };
}

function makeMapModel(
  excludeGguf: boolean,
  excludedTags: ReadonlySet<string>,
  keepUnsupportedTags: boolean,
  idSuffix: string,
  deviceType: string | null,
) {
  const suffixLower = idSuffix.toLowerCase();
  return (raw: unknown): HfModelResult | null => {
    const m = raw as {
      name: string;
      downloads?: number;
      likes?: number;
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
      isGgufLike(m.name);
    if (excludeGguf && isGguf) {
      return null;
    }
    const pipelineTag = m.task ?? m.pipeline_tag;
    const quantMethod = m.config?.quantization_config?.quant_method;
    // Hub-side filtering (Train picker / channel-scoped views) drops models
    // the Unsloth runtime cannot load before they ever reach the row list.
    // The Hub Discover surface opts out via `keepUnsupportedTags=true` so it
    // can still render them with a "may not be supported" dot. Embeddings
    // skip the gate the same way the raw-tag filter does — their pipeline
    // tag (`feature-extraction`/`sentence-similarity`) lives in the
    // unsupported set for chat, but they are explicitly trainable.
    if (!keepUnsupportedTags && !isEmbedding) {
      const support = classifyUnslothSupport({
        modelId: m.name,
        pipelineTag,
        tags: m.tags,
        libraryName: m.library_name,
        deviceType,
        quantMethod,
      });
      if (support.status === "unsupported") return null;
    }
    const updatedAtIso =
      m.updatedAt instanceof Date
        ? m.updatedAt.toISOString()
        : typeof m.updatedAt === "string"
          ? m.updatedAt
          : undefined;
    return {
      id: m.name,
      downloads: m.downloads ?? 0,
      likes: m.likes ?? 0,
      totalParams: m.safetensors?.total ?? m.gguf?.total,
      estimatedSizeBytes: estimateSizeFromDtypes(m.safetensors?.parameters),
      isGguf,
      tags: m.tags,
      pipelineTag,
      updatedAt: updatedAtIso,
      libraryName: m.library_name,
      quantMethod,
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
  task?: HfTaskFilter,
  accessToken?: string,
  pinnedId?: string,
  sortBy: HfSortKey = "downloads",
  direction: HfSortDirection = "desc",
  signal?: AbortSignal,
): AsyncGenerator<unknown> {
  const sortFetch = makeSortFetch(sortBy, direction, signal);
  const tasks = normalizeTaskFilter(task);
  const common = {
    additionalFields: ALL_FIELDS,
    fetch: sortFetch,
    ...(accessToken ? { credentials: { accessToken } } : {}),
  };

  const unslothIter = mergeTaskIterators(tasks, (task) =>
    listModels({
      search: { query, owner: "unsloth", ...(task ? { task } : {}) },
      ...common,
    }) as AsyncGenerator<unknown>,
  );
  const generalIter = mergeTaskIterators(tasks, (task) =>
    listModels({
      search: { query, ...(task ? { task } : {}) },
      ...common,
    }) as AsyncGenerator<unknown>,
  );

  // Start pinned model lookup immediately so it can run in parallel with
  // the Phase 1 unsloth iteration instead of blocking Phase 2.
  const pinnedPromise = pinnedId
    ? cachedModelInfo({
        name: pinnedId,
        additionalFields: ALL_FIELDS,
        fetch: makeHfFetch(signal),
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
  task?: HfTaskFilter,
  accessToken?: string,
  sortBy: HfSortKey = "downloads",
  direction: HfSortDirection = "desc",
  signal?: AbortSignal,
): AsyncGenerator<unknown> {
  const tasks = normalizeTaskFilter(task);
  const common = {
    additionalFields: ALL_FIELDS,
    fetch: makeSortFetch(sortBy, direction, signal),
    ...(accessToken ? { credentials: { accessToken } } : {}),
  };

  // Phase 1: fetch priority models in parallel via modelInfo
  const seen = new Set<string>();
  const settled = await Promise.allSettled(
    priorityIds.map((id) =>
      cachedModelInfo({
        name: id,
        additionalFields: ALL_FIELDS,
        fetch: makeHfFetch(signal),
        ...(accessToken ? { credentials: { accessToken } } : {}),
      }),
    ),
  );
  for (const result of settled) {
    if (result.status === "fulfilled") {
      const m = result.value as { name?: string; pipeline_tag?: string };
      if (!taskMatches(m.pipeline_tag, tasks)) continue;
      if (m.name) seen.add(m.name);
      yield result.value;
    }
  }

  const generalIter = mergeTaskIterators(tasks, (task) =>
    listModels({
      search: { owner: "unsloth", ...(task ? { task } : {}) },
      ...common,
    }) as AsyncGenerator<unknown>,
  );
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
    task?: HfTaskFilter;
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
    (signal: AbortSignal) => {
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
          fetch: makeSortFetch(sortBy, sortDirection, signal),
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
            signal,
          ) as AsyncGenerator<unknown>;
        }
        return mergeTaskIterators(normalizeTaskFilter(task), (task) =>
          listModels({
            search: { ...(task ? { task } : {}) },
            additionalFields: ALL_FIELDS,
            fetch: makeSortFetch(sortBy, sortDirection, signal),
            sort: sortBy,
            ...(accessToken ? { credentials: { accessToken } } : {}),
          }) as AsyncGenerator<unknown>,
        );
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
        signal,
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
  const excludedTags = excludedFormatTagsForDevice(deviceType);
  const mapModel = useMemo(
    () =>
      makeMapModel(
        excludeGguf,
        excludedTags,
        keepUnsupportedTags,
        channelIdSuffix,
        deviceType,
      ),
    [excludeGguf, excludedTags, keepUnsupportedTags, channelIdSuffix, deviceType],
  );
  const search = useHfPaginatedSearch(createIter, mapModel, { enabled });

  // Secondary sort: only when there's no user query. With a query, the merged
  // iterator already floats a small number of unsloth results to the top —
  // re-sorting all loaded results would bury the user's actual search matches.
  // Channel scoping has its own ordering, so the re-sort is also disabled there.
  //
  // STABLE-APPEND CONTRACT: when a later page lands (search.results grew),
  // we keep the previously-sorted prefix verbatim and append only the new tail
  // in its natural order. Re-sorting the merged array would let a freshly
  // arrived unsloth/* repo slip into an earlier index, shifting every other
  // item forward and visibly bumping the user's viewport during infinite
  // scroll. Sorting is therefore applied ONLY when the listing resets (length
  // shrinks or zeros), which is when re-ordering is safe (the user has scrolled
  // back to the top or initiated a new search).
  //
  const channelActive = Boolean(channelOwner || channelTagsKey);
  const [stableCache, setStableCache] = useState<{
    source: HfModelResult[] | null;
    length: number;
    results: HfModelResult[];
    sorted: boolean;
  }>({ source: null, length: 0, results: [], sorted: false });

  const incoming = search.results;
  const sortingDisabled =
    !pinUnslothFirst || isPublisherQuery || trimmed || channelActive;
  let results: HfModelResult[];
  let nextCache = stableCache;

  if (sortingDisabled) {
    results = incoming;
    if (stableCache.results !== incoming || stableCache.sorted) {
      nextCache = {
        source: incoming,
        length: incoming.length,
        results: incoming,
        sorted: false,
      };
    }
  } else if (incoming.length === 0) {
    results = incoming;
    if (
      stableCache.length !== 0 ||
      stableCache.results !== incoming ||
      !stableCache.sorted
    ) {
      nextCache = {
        source: incoming,
        length: 0,
        results: incoming,
        sorted: true,
      };
    }
  } else if (
    !stableCache.sorted ||
    stableCache.length === 0 ||
    incoming.length < stableCache.length ||
    (incoming.length === stableCache.length && stableCache.source !== incoming)
  ) {
    // Listing reset / shrunk / replaced — safe to re-sort from scratch.
    const sorted = [...incoming].sort((a, b) => {
      const aFirst = a.id.startsWith("unsloth/") ? 0 : 1;
      const bFirst = b.id.startsWith("unsloth/") ? 0 : 1;
      return aFirst - bFirst;
    });
    results = sorted;
    nextCache = {
      source: incoming,
      length: incoming.length,
      results: sorted,
      sorted: true,
    };
  } else if (incoming.length === stableCache.length) {
    // No new pages since the cache was last produced.
    results = stableCache.results;
  } else {
    // Listing grew — keep the prior sorted prefix and append the new tail in
    // its natural order so existing virtualized rows don't shift index.
    const newTail = incoming.slice(stableCache.length);
    const merged = stableCache.results.concat(newTail);
    results = merged;
    nextCache = {
      source: incoming,
      length: incoming.length,
      results: merged,
      sorted: true,
    };
  }

  const cacheNeedsUpdate = nextCache !== stableCache;
  useEffect(() => {
    if (!cacheNeedsUpdate) return;
    const handle = globalThis.setTimeout(() => {
      startTransition(() => {
        setStableCache(nextCache);
      });
    }, 0);
    return () => globalThis.clearTimeout(handle);
  }, [cacheNeedsUpdate, nextCache]);

  return { ...search, results };
}
