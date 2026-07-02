// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import type { PipelineType } from "@huggingface/hub";
import { listModels } from "@huggingface/hub";
import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  type CachedResult,
  cachedModelInfo,
  primeCacheFromListing,
} from "../lib/hf-cache";
import { EMBEDDING_TAGS, estimateSizeFromDtypes } from "../lib/hf-model-meta";
import { detectBaseModel } from "../lib/model-capabilities";
import { isGgufLike } from "../lib/model-identifiers";
import { fetchWithTimeout } from "../lib/network";
import {
  type UnslothSupport,
  type UnslothSupportStatus,
  classifyUnslothSupport,
  excludedFormatTagsForDevice,
} from "../lib/unsloth-support";
import { pullBatch, useHubPaginatedSearch } from "./use-hub-paginated-search";

// "gguf" is not in the @huggingface/hub expandable-key type, but the listing
// supports expand=gguf (see withGgufExpand) and listModels' pick() copies any
// requested field through at runtime. Request it here so GGUF repos whose id
// has no "<n>B" token (Kimi, MiniMax, GLM) still populate m.gguf.total for the
// param chip / OOM badge. The cast bridges that single library type gap.
const ALL_FIELDS = [
  "safetensors",
  "tags",
  "library_name",
  "config",
  "createdAt",
  "downloadsAllTime",
  "gguf",
] as unknown as (
  | "safetensors"
  | "tags"
  | "library_name"
  | "config"
  | "createdAt"
  | "downloadsAllTime"
)[];

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
  private?: boolean;
  gated?: false | "auto" | "manual";
  totalParams?: number;
  estimatedSizeBytes?: number;
  isGguf: boolean;
  baseModel?: string | null;
  tags?: string[];
  pipelineTag?: string;
  updatedAt?: string;
  createdAt?: string;
  downloadsAllTime?: number;
  libraryName?: string;
  quantMethod?: string;
}

// HF rejects direction=1 (asc) for trendingScore — only descending is supported.
const DESC_ONLY_SORTS = new Set<HfSortKey>(["trendingScore"]);
const HF_SEARCH_TIMEOUT_MS = 15_000;

// The HF listModels lib doesn't whitelist gguf metadata, but the listing
// supports it. Append expand=gguf so GGUF repos report a param count (used for
// the size / OOM badge) even when the name has no "<n>B" token (Kimi, MiniMax,
// GLM). Shared by every fetch path so the badge is consistent.
function withGgufExpand(input: Parameters<typeof fetch>[0]): string {
  const rawUrl =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;
  const url = new URL(rawUrl);
  if (!url.searchParams.getAll("expand").includes("gguf")) {
    url.searchParams.append("expand", "gguf");
  }
  return url.toString();
}

function makeHfFetch(signal?: AbortSignal): typeof fetch {
  return (input, init) =>
    fetchWithTimeout(
      withGgufExpand(input),
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
      withGgufExpand(url),
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
      private?: boolean;
      gated?: false | "auto" | "manual";
      task?: string;
      pipeline_tag?: string;
      library_name?: string;
      updatedAt?: Date | string;
      createdAt?: Date | string;
      downloadsAllTime?: number;
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
    // Drop runtime-unloadable models before they reach the row list. Discover
    // opts out via keepUnsupportedTags to render them with a "may not be supported"
    // dot. Embeddings skip the gate: their pipeline tag is unsupported for chat but trainable.
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
    const createdAtIso =
      m.createdAt instanceof Date
        ? m.createdAt.toISOString()
        : typeof m.createdAt === "string"
          ? m.createdAt
          : undefined;
    return {
      id: m.name,
      downloads: m.downloads ?? 0,
      likes: m.likes ?? 0,
      private: m.private,
      gated: m.gated,
      totalParams: m.safetensors?.total ?? m.gguf?.total,
      estimatedSizeBytes: estimateSizeFromDtypes(m.safetensors?.parameters),
      isGguf,
      baseModel: detectBaseModel(m.tags),
      tags: m.tags,
      pipelineTag,
      updatedAt: updatedAtIso,
      createdAt: createdAtIso,
      downloadsAllTime: m.downloadsAllTime,
      libraryName: m.library_name,
      quantMethod,
    };
  };
}

/** Unsloth results pulled up-front before yielding general results. */
const UNSLOTH_PREFETCH = 20;
/** With a typed query, float only a few unsloth results before the general listing. */
const UNSLOTH_QUERY_PREFETCH = 3;
/** With a publisher query, fewer unsloth results before the pinned publisher model. */
const UNSLOTH_PINNED_PREFETCH = 4;
/** Matches a valid "owner/repo" identifier (exactly two non-empty segments). */
const PUBLISHER_RE = /^([^/\s]+)\/([^/\s]+)$/;

/**
 * Prime the hf-cache from a listModels result. For public models also prime the
 * anonymous slot so the VRAM hook gets cache hits; gated/private models are cached
 * only under the caller's token to avoid auth leakage.
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

/** Merged generator yielding unsloth-owned models first, then deduped general results. */
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

  const unslothIter = mergeTaskIterators(
    tasks,
    (task) =>
      listModels({
        search: { query, owner: "unsloth", ...(task ? { task } : {}) },
        ...common,
      }) as AsyncGenerator<unknown>,
  );
  const generalIter = mergeTaskIterators(
    tasks,
    (task) =>
      listModels({
        search: { query, ...(task ? { task } : {}) },
        ...common,
      }) as AsyncGenerator<unknown>,
  );

  // Start the pinned lookup now so it runs in parallel with Phase 1 instead of blocking Phase 2.
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

  // Phase 1: unsloth models first
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

  // Phase 1b: pinned publisher model before general results
  if (pinnedId && !seen.has(pinnedId) && pinnedPromise) {
    const pinned = await pinnedPromise;
    if (pinned) {
      // Record raw input and HF's canonical name so Phase 2 dedup survives casing differences.
      seen.add(pinnedId);
      const canonicalName = (pinned as { name?: string }).name;
      if (canonicalName && canonicalName !== pinnedId) {
        seen.add(canonicalName);
      }
      yield pinned;
    }
  }

  // Phase 2: general results, skipping already-seen models
  for await (const model of generalIter) {
    const m = model as { name?: string };
    if (m.name && seen.has(m.name)) continue;
    if (m.name) {
      primeFromListing(m.name, accessToken, model);
    }
    yield model;
  }
}

/** Yields priority models (fetched individually for full metadata), then the unsloth listing. */
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

  // Phase 1: priority models in parallel via modelInfo
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

  const generalIter = mergeTaskIterators(
    tasks,
    (task) =>
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

function createChannelIterator(
  channel: HfModelSearchChannel,
  opts: {
    query?: string;
    sortBy: HfSortKey;
    sortDirection: HfSortDirection;
    accessToken?: string;
    signal: AbortSignal;
  },
): AsyncGenerator<unknown> {
  const channelTags =
    channel.tags && channel.tags.length ? [...channel.tags] : undefined;
  const queryString = opts.query || channel.query || undefined;
  return listModels({
    search: {
      ...(queryString ? { query: queryString } : {}),
      ...(channel.owner ? { owner: channel.owner } : {}),
      ...(channelTags ? { tags: channelTags } : {}),
    },
    additionalFields: ALL_FIELDS,
    fetch: makeSortFetch(opts.sortBy, opts.sortDirection, opts.signal),
    sort: opts.sortBy,
    ...(opts.accessToken
      ? { credentials: { accessToken: opts.accessToken } }
      : {}),
  }) as AsyncGenerator<unknown>;
}

// Bound the unsloth pass so a huge unsloth slice can't starve the general listing under scroll.
const UNSLOTH_CHANNEL_PREFETCH = 60;

// For tag/format channels without a fixed owner (e.g. GGUF filter), yield unsloth
// models first (in sort order), then the rest with already-seen unsloth repos removed,
// floating unsloth to the top even when sorting by downloads/likes.
async function* channelUnslothFirstIterator(
  channel: { tags?: string[]; query?: string },
  opts: {
    query?: string;
    sortBy: HfSortKey;
    sortDirection: HfSortDirection;
    accessToken?: string;
    signal: AbortSignal;
  },
): AsyncGenerator<unknown> {
  const queryString = opts.query || channel.query || undefined;
  const creds = opts.accessToken
    ? { credentials: { accessToken: opts.accessToken } }
    : {};
  const seen = new Set<string>();

  const unslothIter = listModels({
    search: {
      ...(queryString ? { query: queryString } : {}),
      owner: "unsloth",
      ...(channel.tags ? { tags: channel.tags } : {}),
    },
    additionalFields: ALL_FIELDS,
    fetch: makeSortFetch(opts.sortBy, opts.sortDirection, opts.signal),
    sort: opts.sortBy,
    ...creds,
  }) as AsyncGenerator<unknown>;
  let count = 0;
  for await (const model of unslothIter) {
    const name = (model as { name?: string }).name;
    if (name) seen.add(name);
    yield model;
    if (++count >= UNSLOTH_CHANNEL_PREFETCH) break;
  }

  const generalIter = listModels({
    search: {
      ...(queryString ? { query: queryString } : {}),
      ...(channel.tags ? { tags: channel.tags } : {}),
    },
    additionalFields: ALL_FIELDS,
    fetch: makeSortFetch(opts.sortBy, opts.sortDirection, opts.signal),
    sort: opts.sortBy,
    ...creds,
  }) as AsyncGenerator<unknown>;
  for await (const model of generalIter) {
    const name = (model as { name?: string }).name;
    if (name && seen.has(name)) continue;
    yield model;
  }
}

export interface FetchChannelFirstPageOptions {
  channel: HfModelSearchChannel;
  sortBy: HfSortKey;
  sortDirection?: HfSortDirection;
  accessToken?: string;
  signal: AbortSignal;
  pageSize?: number;
  deviceType: string | null;
  excludeGguf?: boolean;
  keepUnsupportedTags?: boolean;
}

export interface FetchChannelFirstPageResult {
  results: HfModelResult[];
  scanned: number;
  done: boolean;
}

export async function fetchChannelFirstPage(
  options: FetchChannelFirstPageOptions,
): Promise<FetchChannelFirstPageResult> {
  const {
    channel,
    sortBy,
    sortDirection = "desc",
    accessToken,
    signal,
    pageSize = 20,
    deviceType,
    excludeGguf = false,
    keepUnsupportedTags = true,
  } = options;
  const mapModel = makeMapModel(
    excludeGguf,
    excludedFormatTagsForDevice(deviceType),
    keepUnsupportedTags,
    channel.idSuffix ?? "",
    deviceType,
  );
  async function* primed(): AsyncGenerator<unknown> {
    const iter = createChannelIterator(channel, {
      sortBy,
      sortDirection,
      accessToken,
      signal,
    });
    for await (const model of iter) {
      const name = (model as { name?: string }).name;
      if (name) primeFromListing(name, accessToken, model);
      yield model;
    }
  }
  const { items, done, scanned } = await pullBatch(
    primed(),
    mapModel,
    pageSize,
  );
  return { results: items, scanned, done };
}

export function useHubModelSearch(
  query: string,
  options?: {
    task?: HfTaskFilter;
    accessToken?: string;
    excludeGguf?: boolean;
    priorityIds?: readonly string[];
    sortBy?: HfSortKey;
    sortDirection?: HfSortDirection;
    pinUnslothFirst?: boolean;
    /**
     * "unsloth" restricts listings to the unsloth org; "all" surfaces the whole
     * Hub with unsloth floated to the top. Owner-fixed channel presets ignore this.
     */
    ownerScope?: "unsloth" | "all";
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
    ownerScope = "all",
    enabled = true,
    keepUnsupportedTags = false,
    channel = null,
  } = options ?? {};
  const unslothOnly = ownerScope === "unsloth";

  const channelOwner = channel?.owner ?? null;
  const channelTagsKey = channel?.tags ? channel.tags.join("|") : "";
  const channelQuery = channel?.query ?? "";
  const channelIdSuffix = channel?.idSuffix ?? "";
  const priorityIdsKey = priorityIds?.join("|") ?? "";
  const stablePriorityIds = useMemo(
    () => (priorityIdsKey ? priorityIdsKey.split("|") : undefined),
    [priorityIdsKey],
  );

  // Parse publisher detection once, shared between the iterator factory and the secondary sort gate.
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
      // Channel scoping bypasses the unsloth-merge iterator: a hard owner/tag
      // filter shows just that curated slice, with the user's text query forwarded in.
      if (channelOwner || channelTagsKey || channelQuery) {
        const channelTags = channelTagsKey
          ? channelTagsKey.split("|")
          : undefined;
        // Unsloth-only scope on an ownerless tag/format channel (e.g. GGUF filter):
        // hard-restrict the slice to unsloth-owned repos.
        if (unslothOnly && !channelOwner) {
          return createChannelIterator(
            {
              owner: "unsloth",
              tags: channelTags,
              query: channelQuery || undefined,
            },
            {
              query: trimmed || undefined,
              sortBy,
              sortDirection,
              accessToken,
              signal,
            },
          );
        }
        // Ownerless tag/format channels (e.g. GGUF filter): float unsloth-owned models first.
        if (pinUnslothFirst && channelTagsKey && !channelOwner) {
          return channelUnslothFirstIterator(
            { tags: channelTags, query: channelQuery || undefined },
            {
              query: trimmed || undefined,
              sortBy,
              sortDirection,
              accessToken,
              signal,
            },
          );
        }
        // User text query takes precedence over channel.query; without it, the
        // channel's own query (e.g. "bnb-4bit") narrows the listing server-side.
        return createChannelIterator(
          {
            owner: channelOwner ?? undefined,
            tags: channelTags,
            query: channelQuery || undefined,
          },
          {
            query: trimmed || undefined,
            sortBy,
            sortDirection,
            accessToken,
            signal,
          },
        );
      }
      if (!trimmed) {
        // No query: show priority models first (with full metadata), then general unsloth listing
        if (stablePriorityIds && stablePriorityIds.length > 0) {
          return priorityThenListingIterator(
            stablePriorityIds,
            task,
            accessToken,
            sortBy,
            sortDirection,
            signal,
          ) as AsyncGenerator<unknown>;
        }
        return mergeTaskIterators(
          normalizeTaskFilter(task),
          (task) =>
            listModels({
              // Unsloth-only scope restricts the plain sort browse to the org.
              search: {
                ...(unslothOnly ? { owner: "unsloth" } : {}),
                ...(task ? { task } : {}),
              },
              additionalFields: ALL_FIELDS,
              fetch: makeSortFetch(sortBy, sortDirection, signal),
              sort: sortBy,
              ...(accessToken ? { credentials: { accessToken } } : {}),
            }) as AsyncGenerator<unknown>,
        );
      }
      // Unsloth-only typed query: search within the org rather than floating
      // a few unsloth hits above the global relevance ranking.
      if (unslothOnly) {
        return listModels({
          search: { query: searchQuery, owner: "unsloth" },
          additionalFields: ALL_FIELDS,
          fetch: makeSortFetch(sortBy, sortDirection, signal),
          sort: sortBy,
          ...(accessToken ? { credentials: { accessToken } } : {}),
        }) as AsyncGenerator<unknown>;
      }
      // Typed query: drop the task filter so searched models appear despite
      // wrong/missing HF task metadata. For an "owner/repo" query, strip the org
      // prefix so unsloth variants surface, then pin the original publisher model.
      // Unsloth-owned queries are left as-is for the full prefetch + secondary sort.
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
      stablePriorityIds,
      sortBy,
      sortDirection,
      channelOwner,
      channelTagsKey,
      channelQuery,
      pinUnslothFirst,
      unslothOnly,
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
    [
      excludeGguf,
      excludedTags,
      keepUnsupportedTags,
      channelIdSuffix,
      deviceType,
    ],
  );
  const search = useHubPaginatedSearch(createIter, mapModel, { enabled });

  // Secondary sort only with no user query (merged iterator already floats
  // unsloth results; re-sorting would bury matches) and outside channel scoping.
  //
  // STABLE-APPEND CONTRACT: when a later page lands, keep the sorted prefix
  // verbatim and append only the new tail. Re-sorting the whole array would let
  // a late unsloth/* repo jump to an earlier index and bump the viewport during
  // infinite scroll, so we only sort when the listing resets (length shrinks or
  // zeros), where re-ordering is safe.
  const [stableCache, setStableCache] = useState<{
    source: HfModelResult[] | null;
    length: number;
    results: HfModelResult[];
    sorted: boolean;
  }>({ source: null, length: 0, results: [], sorted: false });

  const incoming = search.results;
  // Owner-scoped channels return a single owner, so re-sorting is a no-op there;
  // tag/format channels (e.g. GGUF) and plain browsing still float unsloth first.
  const sortingDisabled =
    !pinUnslothFirst || isPublisherQuery || trimmed || Boolean(channelOwner);
  const { results, nextCache } = useMemo(() => {
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
      (incoming.length === stableCache.length &&
        stableCache.source !== incoming)
    ) {
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
      results = stableCache.results;
    } else {
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

    return { results, nextCache };
  }, [incoming, sortingDisabled, stableCache]);

  const cacheNeedsUpdate = nextCache !== stableCache;
  useEffect(() => {
    if (!cacheNeedsUpdate) return;
    startTransition(() => {
      setStableCache((current) =>
        current === stableCache ? nextCache : current,
      );
    });
  }, [cacheNeedsUpdate, nextCache, stableCache]);

  return { ...search, results };
}
