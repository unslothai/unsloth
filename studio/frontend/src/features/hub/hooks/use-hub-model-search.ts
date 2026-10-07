// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { getHfEndpoint, useHfEndpoint } from "@/lib/hf-endpoint";
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
import { mergeTaskIterators } from "../lib/merge-task-iterators";
import { detectBaseModel } from "../lib/model-capabilities";
import { isGgufLike } from "../lib/model-identifiers";
import { fetchWithTimeout } from "../lib/network";
import {
  type UnslothSupport,
  type UnslothSupportStatus,
  classifyUnslothSupport,
} from "../lib/unsloth-support";
import { pullBatch, useHubPaginatedSearch } from "./use-hub-paginated-search";

// "gguf" is not in the SDK's expandable-key type, but listModels passes it through at runtime;
// it fills m.gguf.total for repos whose id has no "<n>B" token.
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

export interface HfModelResult {
  id: string;
  downloads: number;
  likes: number;
  private?: boolean;
  gated?: false | "auto" | "manual";
  totalParams?: number;
  estimatedSizeBytes?: number;
  curatedSizeBytes?: number;
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

// HF rejects ascending order for trendingScore.
const DESC_ONLY_SORTS = new Set<HfSortKey>(["trendingScore"]);
const HF_SEARCH_TIMEOUT_MS = 15_000;

// Append expand=gguf so GGUF repos report a param count even without a "<n>B" token.
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
    // A "gguf"-tagged diffusers pipeline ships no .gguf files; trust the bare tag only for non-pipelines.
    const isDiffusersPipeline =
      m.library_name?.toLowerCase() === "diffusers" ||
      Boolean(m.tags?.some((tag) => tag.toLowerCase().startsWith("diffusers:")));
    const isGguf =
      isGgufLike(m.name) ||
      Boolean(m.gguf) ||
      (Boolean(m.tags?.some((tag) => tag.toLowerCase() === "gguf")) &&
        !isDiffusersPipeline);
    if (excludeGguf && isGguf) {
      return null;
    }
    const pipelineTag = m.task ?? m.pipeline_tag;
    const quantMethod = m.config?.quantization_config?.quant_method;
    // Embeddings skip the gate: unsupported for chat but trainable.
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

const UNSLOTH_PREFETCH = 20;
const UNSLOTH_QUERY_PREFETCH = 3;
const UNSLOTH_PINNED_PREFETCH = 4;
const PUBLISHER_RE = /^([^/\s]+)\/([^/\s]+)$/;

/** Public models also prime the anonymous slot; gated/private only under the caller's token. */
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

async function* mergedModelIterator(
  query: string,
  task?: HfTaskFilter,
  accessToken?: string,
  pinnedId?: string,
  sortBy: HfSortKey = "downloads",
  direction: HfSortDirection = "desc",
  signal?: AbortSignal,
): AsyncGenerator<unknown> {
  const tasks = normalizeTaskFilter(task);
  const common = {
    additionalFields: ALL_FIELDS,
    ...(accessToken ? { credentials: { accessToken } } : {}),
  };

  const unslothIter = mergeTaskIterators(
    tasks,
    (task, taskSignal) =>
      listModels({
        hubUrl: getHfEndpoint(),
        search: { query, owner: "unsloth", ...(task ? { task } : {}) },
        fetch: makeSortFetch(sortBy, direction, taskSignal),
        ...common,
      }) as AsyncGenerator<unknown>,
    signal,
  );
  const generalIter = mergeTaskIterators(
    tasks,
    (task, taskSignal) =>
      listModels({
        hubUrl: getHfEndpoint(),
        search: { query, ...(task ? { task } : {}) },
        fetch: makeSortFetch(sortBy, direction, taskSignal),
        ...common,
      }) as AsyncGenerator<unknown>,
    signal,
  );

  // Start the pinned lookup now so it runs in parallel with Phase 1.
  const pinnedPromise = pinnedId
    ? cachedModelInfo({
        hubUrl: getHfEndpoint(),
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
      // Record raw input and HF's canonical name so dedup survives casing differences.
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
    ...(accessToken ? { credentials: { accessToken } } : {}),
  };

  // Phase 1: priority models in parallel via modelInfo
  const seen = new Set<string>();
  const settled = await Promise.allSettled(
    priorityIds.map((id) =>
      cachedModelInfo({
        hubUrl: getHfEndpoint(),
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
    (task, taskSignal) =>
      listModels({
        hubUrl: getHfEndpoint(),
        search: { owner: "unsloth", ...(task ? { task } : {}) },
        fetch: makeSortFetch(sortBy, direction, taskSignal),
        ...common,
      }) as AsyncGenerator<unknown>,
    signal,
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
  query?: string;
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
    hubUrl: getHfEndpoint(),
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

// Bounded so a huge unsloth slice cannot starve the general listing.
const UNSLOTH_CHANNEL_PREFETCH = 60;

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
    hubUrl: getHfEndpoint(),
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
    hubUrl: getHfEndpoint(),
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
    /** "all" floats unsloth to the top; owner-fixed channel presets ignore this. */
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

  const hfEndpoint = useHfEndpoint();
  const createIter = useCallback(
    (signal: AbortSignal) => {
      if (channelOwner || channelTagsKey || channelQuery) {
        const channelTags = channelTagsKey
          ? channelTagsKey.split("|")
          : undefined;
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
          (task, taskSignal) =>
            listModels({
              hubUrl: getHfEndpoint(),
              search: {
                ...(unslothOnly ? { owner: "unsloth" } : {}),
                ...(task ? { task } : {}),
              },
              additionalFields: ALL_FIELDS,
              fetch: makeSortFetch(sortBy, sortDirection, taskSignal),
              sort: sortBy,
              ...(accessToken ? { credentials: { accessToken } } : {}),
            }) as AsyncGenerator<unknown>,
          signal,
        );
      }
      if (unslothOnly) {
        return listModels({
          hubUrl: getHfEndpoint(),
          search: { query: searchQuery, owner: "unsloth" },
          additionalFields: ALL_FIELDS,
          fetch: makeSortFetch(sortBy, sortDirection, signal),
          sort: sortBy,
          ...(accessToken ? { credentials: { accessToken } } : {}),
        }) as AsyncGenerator<unknown>;
      }
      // Typed query: drop the task filter (HF task metadata is unreliable); for "owner/repo" strip the
      // org so unsloth variants surface, then pin the original.
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
      hfEndpoint,
    ],
  );

  const deviceType = usePlatformStore((s) => s.deviceType);
  const mapModel = useMemo(
    () =>
      makeMapModel(
        excludeGguf,
        keepUnsupportedTags,
        channelIdSuffix,
        deviceType,
      ),
    [
      excludeGguf,
      keepUnsupportedTags,
      channelIdSuffix,
      deviceType,
    ],
  );
  const search = useHubPaginatedSearch(createIter, mapModel, { enabled });

  // Stable-append contract: keep the sorted prefix and append only new pages, else late rows jump
  // and bump the viewport. Re-sort only when the listing resets.
  const [stableCache, setStableCache] = useState<{
    source: HfModelResult[] | null;
    length: number;
    results: HfModelResult[];
    sorted: boolean;
  }>({ source: null, length: 0, results: [], sorted: false });

  const incoming = search.results;
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
