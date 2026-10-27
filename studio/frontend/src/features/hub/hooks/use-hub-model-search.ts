// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { getHfEndpoint, useHfEndpoint } from "@/lib/hf-endpoint";
import type { PipelineType } from "@huggingface/hub";
import { listModels, modelInfo } from "@huggingface/hub";
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
import { ggufRepoParams } from "@/lib/model-size";
import { type FreshRankable, rankFresh } from "../lib/fresh-ranking";
import { type ProviderLogo, resolveOwnerProviderLogo } from "../lib/provider-logos";
import {
  matchProviderByQuery,
  providerHubOrgs,
  providerSearchTerms,
} from "../lib/provider-search";
import { fetchWithTimeout } from "../lib/network";
import { LruMap } from "../lib/lru-map";
import {
  type UnslothSupport,
  type UnslothSupportStatus,
  classifyUnslothSupport,
} from "../lib/unsloth-support";
import {
  filterModelListing,
  hasModelSearchFilters,
  needsModelMetadata,
  parameterRange,
  type ModelSearchFilters,
} from "../lib/model-search-filters";
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
  "trendingScore",
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
  // Not a Hub sort: trending weighed against age, ranked here (lib/fresh-ranking.ts).
  | "fresh"
  | "trendingScore"
  | "downloads"
  | "likes"
  | "lastModified"
  | "createdAt";

export type HfSortDirection = "desc" | "asc";

/** The Hub's own sort for a listing; fresh is ranked from a newest-first one. */
export function hubListingSort(sortBy: HfSortKey): Exclude<HfSortKey, "fresh"> {
  return sortBy === "fresh" ? "createdAt" : sortBy;
}
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
const FILTER_METADATA_TTL_MS = 5 * 60 * 1000;

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
  parameterFilter = "",
): typeof fetch {
  return (input, init) => {
    const rawUrl =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
    const url = new URL(rawUrl);
    if (parameterFilter)
      url.searchParams.set("num_parameters", parameterFilter);

    if (sortBy && !url.searchParams.has("sort")) {
      url.searchParams.set("sort", hubListingSort(sortBy));
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
  hideEmptyUploads = false,
) {
  const suffixLower = idSuffix.toLowerCase();
  return (raw: unknown): HfModelResult | null => {
    if (raw === null) return null;
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
    // The Hub derives gguf metadata and the "gguf" tag from uploaded files: neither means none yet.
    if (
      hideEmptyUploads &&
      isGgufLike(m.name) &&
      !m.gguf &&
      !m.tags?.some((tag) => tag.toLowerCase() === "gguf")
    ) {
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
      totalParams:
        m.safetensors?.total ??
        (m.gguf ? ggufRepoParams(m.name, m.gguf.total) : undefined),
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
/** Rejected unsloth rows a filtered search scans before moving on to general results. */
const UNSLOTH_FILTER_REJECT_LIMIT = 40;
/** Matches a valid "owner/repo" identifier (exactly two non-empty segments). */
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

type ModelListingTransform = (
  iterator: AsyncGenerator<unknown>,
) => AsyncGenerator<unknown>;

/** Merged generator yielding unsloth-owned models first, then deduped general results. */
async function* mergedModelIterator(
  query: string,
  task?: HfTaskFilter,
  accessToken?: string,
  pinnedId?: string,
  sortBy: HfSortKey = "downloads",
  direction: HfSortDirection = "desc",
  signal?: AbortSignal,
  parameterFilter = "",
  filterPriority?: ModelListingTransform,
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
        fetch: makeSortFetch(sortBy, direction, taskSignal, parameterFilter),
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
        fetch: makeSortFetch(sortBy, direction, taskSignal, parameterFilter),
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
  let rejected = 0;
  for await (const model of filterPriority
    ? filterPriority(unslothIter)
    : unslothIter) {
    if (model === null) {
      yield null;
      if (++rejected >= UNSLOTH_FILTER_REJECT_LIMIT) break;
      continue;
    }
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
  parameterFilter = "",
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
        fetch: makeSortFetch(sortBy, direction, taskSignal, parameterFilter),
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

/** Unsloth uploads read per search term for a model maker's query. */
const PROVIDER_POOL = 100;

function rankListing(
  models: FreshRankable[],
  sortBy: HfSortKey,
  direction: HfSortDirection,
): FreshRankable[] {
  if (sortBy === "fresh") {
    return rankFresh(models, Date.now());
  }
  const key = (model: FreshRankable & { downloads?: number }) => {
    if (sortBy === "trendingScore") return model.trendingScore ?? 0;
    if (sortBy === "likes") return model.likes ?? 0;
    if (sortBy === "downloads") return model.downloads ?? 0;
    const date = sortBy === "createdAt" ? model.createdAt : model.updatedAt;
    return date ? new Date(date).getTime() : 0;
  };
  const sign = direction === "asc" ? 1 : -1;
  return [...models].sort((a, b) => sign * (key(a) - key(b)));
}

/**
 * A search naming a model maker ("meta", "alibaba"): Unsloth's uploads of that maker's models, the
 * ones shown with its avatar, then (beyond Unsloth-only) the maker's own org uploads. Each group's
 * first pool is ranked by the listing's sort; the rest of each listing follows in its own order,
 * then the search's own results without repeats.
 */
async function* providerUnslothFirstIterator(
  provider: ProviderLogo,
  inner: AsyncGenerator<unknown>,
  opts: {
    sortBy: HfSortKey;
    sortDirection: HfSortDirection;
    accessToken?: string;
    signal: AbortSignal;
    parameterFilter?: string;
    includeOrgs: boolean;
    filterPriority?: ModelListingTransform;
  },
): AsyncGenerator<unknown> {
  const listing = (search: { query?: string; owner: string }) =>
    listModels({
      hubUrl: getHfEndpoint(),
      search,
      additionalFields: ALL_FIELDS,
      fetch: makeSortFetch(opts.sortBy, opts.sortDirection, opts.signal, opts.parameterFilter),
      sort: hubListingSort(opts.sortBy),
      ...(opts.accessToken ? { credentials: { accessToken: opts.accessToken } } : {}),
    }) as AsyncGenerator<unknown>;
  // Read by hand: breaking out of a for-await would close the listing its tail continues.
  const take = async (iter: AsyncGenerator<unknown>) => {
    const items: unknown[] = [];
    try {
      while (items.length < PROVIDER_POOL) {
        const next = await iter.next();
        if (next.done) break;
        items.push(next.value);
      }
    } catch {
      // One listing failing leaves the others; the search's own results still follow.
    }
    return items;
  };
  const drain = async function* (iter: AsyncGenerator<unknown>) {
    try {
      yield* iter;
    } catch {
      // As above: a failed page ends that listing only.
    }
  };
  const nameOf = (model: unknown) => (model as { name?: string } | null)?.name;
  const isProviders = (name: string) =>
    resolveOwnerProviderLogo("unsloth", name.split("/")[1])?.id === provider.id;
  const unslothIters = providerSearchTerms(provider).map((term) =>
    listing({ query: term, owner: "unsloth" }),
  );
  const orgIters = opts.includeOrgs
    ? providerHubOrgs(provider).map((owner) => listing({ owner }))
    : [];
  // Started together; the Unsloth group yields without waiting on the org listings.
  const orgFirst = Promise.all(orgIters.map(take));
  const unslothFirst = await Promise.all(unslothIters.map(take));
  const shown = new Set<string>();
  const ranked = function* (models: unknown[], keep: (name: string) => boolean) {
    const group = new Map<string, FreshRankable>();
    for (const model of models) {
      const name = nameOf(model);
      if (name && !shown.has(name) && !group.has(name) && keep(name)) {
        group.set(name, model as FreshRankable);
      }
    }
    for (const model of rankListing([...group.values()], opts.sortBy, opts.sortDirection)) {
      const name = nameOf(model) as string;
      shown.add(name);
      primeFromListing(name, opts.accessToken, model);
      yield model as unknown;
    }
  };
  const tail = async function* (iters: AsyncGenerator<unknown>[], keep: (name: string) => boolean) {
    for (const iter of iters) {
      for await (const model of drain(iter)) {
        const name = nameOf(model);
        if (!name || shown.has(name) || !keep(name)) continue;
        shown.add(name);
        primeFromListing(name, opts.accessToken, model);
        yield model;
      }
    }
  };
  const prefix = async function* () {
    yield* ranked(unslothFirst.flat(), isProviders);
    yield* ranked((await orgFirst).flat(), () => true);
    yield* tail(unslothIters, isProviders);
    yield* tail(orgIters, () => true);
  };
  // Like the Unsloth pass in mergedModelIterator: a client-side filter that rejects these groups
  // gives way to the search's own results after a bounded scan.
  let rejected = 0;
  for await (const model of opts.filterPriority ? opts.filterPriority(prefix()) : prefix()) {
    if (model === null) {
      yield null;
      if (++rejected >= UNSLOTH_FILTER_REJECT_LIMIT) break;
      continue;
    }
    yield model;
  }
  for await (const model of inner) {
    const name = nameOf(model);
    if (name && shown.has(name)) continue;
    yield model;
  }
}

const FRESH_POOL = 100;

/** Ranks the newest and most trending together, then the rest by date. */
async function* freshChannelIterator(
  channel: HfModelSearchChannel,
  opts: Omit<Parameters<typeof createChannelIterator>[1], "sortBy" | "task"> & {
    tasks?: readonly PipelineType[];
  },
): AsyncGenerator<unknown> {
  const { tasks = [], ...listingOpts } = opts;
  const listing = (sortBy: HfSortKey) =>
    mergeTaskIterators(
      tasks,
      (task, signal) =>
        createChannelIterator(channel, {
          ...listingOpts,
          sortBy,
          sortDirection: "desc",
          signal,
          task,
        }),
      opts.signal,
    );
  const newest = listing("createdAt");
  // Read by hand: breaking out of a for-await would close the listing the tail continues.
  // The task listings alternate, so each task gets a full pool of its own.
  const poolSize = FRESH_POOL * Math.max(1, tasks.length);
  const take = async (iter: AsyncGenerator<unknown>) => {
    const items: unknown[] = [];
    while (items.length < poolSize) {
      const next = await iter.next();
      if (next.done) break;
      items.push(next.value);
    }
    return items;
  };
  const trending = listing("trendingScore");
  const [byDate, byTrend] = await Promise.all([take(newest), take(trending)]);
  void trending.return(undefined);
  const pool = new Map<string, FreshRankable>();
  for (const model of [...byDate, ...byTrend]) {
    const name = (model as { name?: string }).name;
    if (name && !pool.has(name)) pool.set(name, model as FreshRankable);
  }
  yield* rankFresh([...pool.values()], Date.now());
  for await (const model of newest) {
    const name = (model as { name?: string }).name;
    if (!name || !pool.has(name)) yield model;
  }
}

function createChannelIterator(
  channel: HfModelSearchChannel,
  opts: {
    query?: string;
    sortBy: HfSortKey;
    sortDirection: HfSortDirection;
    accessToken?: string;
    signal: AbortSignal;
    parameterFilter?: string;
    task?: PipelineType;
  },
): AsyncGenerator<unknown> {
  if (opts.sortBy === "fresh") {
    return freshChannelIterator(channel, opts);
  }
  const channelTags =
    channel.tags && channel.tags.length ? [...channel.tags] : undefined;
  const queryString = opts.query || channel.query || undefined;
  return listModels({
    hubUrl: getHfEndpoint(),
    search: {
      ...(queryString ? { query: queryString } : {}),
      ...(channel.owner ? { owner: channel.owner } : {}),
      ...(channelTags ? { tags: channelTags } : {}),
      ...(opts.task ? { task: opts.task } : {}),
    },
    additionalFields: ALL_FIELDS,
    fetch: makeSortFetch(
      opts.sortBy,
      opts.sortDirection,
      opts.signal,
      opts.parameterFilter,
    ),
    sort: hubListingSort(opts.sortBy),
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
    parameterFilter?: string;
    filterPriority?: ModelListingTransform;
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
    fetch: makeSortFetch(
      opts.sortBy,
      opts.sortDirection,
      opts.signal,
      opts.parameterFilter,
    ),
    sort: hubListingSort(opts.sortBy),
    ...creds,
  }) as AsyncGenerator<unknown>;
  let count = 0;
  let rejected = 0;
  for await (const model of opts.filterPriority
    ? opts.filterPriority(unslothIter)
    : unslothIter) {
    if (model === null) {
      yield null;
      if (++rejected >= UNSLOTH_FILTER_REJECT_LIMIT) break;
      continue;
    }
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
    fetch: makeSortFetch(
      opts.sortBy,
      opts.sortDirection,
      opts.signal,
      opts.parameterFilter,
    ),
    sort: hubListingSort(opts.sortBy),
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
    /** Drop "-GGUF" repos with no GGUF files uploaded yet. The Hub page keeps them listed. */
    hideEmptyUploads?: boolean;
    channel?: HfModelSearchChannel | null;
    filters?: ModelSearchFilters;
    /** False for a result the caller hides anyway; such rows skip per-row filter lookups. */
    keepResult?: (result: HfModelResult) => boolean;
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
    hideEmptyUploads = false,
    channel = null,
    filters,
    keepResult,
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

  // An owner/repo query names one repo; only a bare maker name widens to its Unsloth uploads.
  const queryProvider = useMemo(
    () => (isPublisherQuery || !trimmed ? null : matchProviderByQuery(trimmed)),
    [isPublisherQuery, trimmed],
  );
  const hfEndpoint = useHfEndpoint();
  // Filter changes and retries reuse settled metadata; another Hub or token must not.
  const filterMetadata = useMemo(
    () =>
      new LruMap<
        string,
        { data: Record<string, unknown> | null; expires: number }
      >(2048),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [hfEndpoint, accessToken],
  );
  const hasFilters = !!filters && hasModelSearchFilters(filters);
  const filterRange = filters ? parameterRange(filters) : "";
  const createListing = useCallback(
    (signal: AbortSignal, filterPriority?: ModelListingTransform) => {
      // Channel scoping bypasses the unsloth-merge iterator: a hard owner/tag filter shows that slice.
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
              parameterFilter: filterRange,
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
              parameterFilter: filterRange,
              filterPriority,
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
            parameterFilter: filterRange,
          },
        );
      }
      // Browsing only: a typed search keeps the search paths below (Unsloth floated, tasks checked client-side).
      if (sortBy === "fresh" && !trimmed) {
        return freshChannelIterator(
          { owner: unslothOnly ? "unsloth" : undefined },
          {
            sortDirection,
            accessToken,
            signal,
            parameterFilter: filterRange,
            tasks: normalizeTaskFilter(task),
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
            filterRange,
          ) as AsyncGenerator<unknown>;
        }
        if (hasFilters && pinUnslothFirst && !unslothOnly) {
          return mergedModelIterator(
            "",
            task,
            accessToken,
            undefined,
            sortBy,
            sortDirection,
            signal,
            filterRange,
            filterPriority,
          );
        }
        return mergeTaskIterators(
          normalizeTaskFilter(task),
          (task, taskSignal) =>
            listModels({
              hubUrl: getHfEndpoint(),
              // Unsloth-only scope restricts the plain sort browse to the org.
              search: {
                ...(unslothOnly ? { owner: "unsloth" } : {}),
                ...(task ? { task } : {}),
              },
              additionalFields: ALL_FIELDS,
              fetch: makeSortFetch(
                sortBy,
                sortDirection,
                taskSignal,
                filterRange,
              ),
              sort: hubListingSort(sortBy),
              ...(accessToken ? { credentials: { accessToken } } : {}),
            }) as AsyncGenerator<unknown>,
          signal,
        );
      }
      const providerOpts = {
        sortBy,
        sortDirection,
        accessToken,
        signal,
        parameterFilter: filterRange,
      };
      // Unsloth-only typed query: search within the org rather than floating a few hits globally.
      if (unslothOnly) {
        const orgSearch = listModels({
          hubUrl: getHfEndpoint(),
          search: { query: searchQuery, owner: "unsloth" },
          additionalFields: ALL_FIELDS,
          fetch: makeSortFetch(sortBy, sortDirection, signal, filterRange),
          sort: hubListingSort(sortBy),
          ...(accessToken ? { credentials: { accessToken } } : {}),
        }) as AsyncGenerator<unknown>;
        return queryProvider
          ? providerUnslothFirstIterator(queryProvider, orgSearch, {
              ...providerOpts,
              includeOrgs: false,
              filterPriority,
            })
          : orgSearch;
      }
      // Typed query: drop the task filter so searched models appear despite wrong/missing HF task
      // metadata. For an "owner/repo" query, strip the org prefix so unsloth variants surface, then
      // pin the original publisher model. Unsloth-owned queries are left as-is.
      const merged = mergedModelIterator(
        searchQuery,
        undefined,
        accessToken,
        hasFilters ? undefined : pinnedId,
        sortBy,
        sortDirection,
        signal,
        filterRange,
        filterPriority,
      ) as AsyncGenerator<unknown>;
      return queryProvider
        ? providerUnslothFirstIterator(queryProvider, merged, {
            ...providerOpts,
            includeOrgs: true,
            filterPriority,
          })
        : merged;
    },
    [
      trimmed,
      searchQuery,
      pinnedId,
      queryProvider,
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
      hasFilters,
      filterRange,
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
        hideEmptyUploads,
      ),
    [
      excludeGguf,
      keepUnsupportedTags,
      channelIdSuffix,
      deviceType,
      hideEmptyUploads,
    ],
  );
  const needsMetadata = !!filters && needsModelMetadata(filters);
  // Only a row the list can show is worth a per-row Hub request.
  const keepRow = useMemo(
    () =>
      needsMetadata
        ? (raw: unknown) => {
            try {
              const result = mapModel(raw);
              return result !== null && (keepResult?.(result) ?? true);
            } catch {
              // Pagination skips a row its mapper throws on; skip it here too, not the search.
              return false;
            }
          }
        : undefined,
    [needsMetadata, mapModel, keepResult],
  );

  const createIter = useCallback(
    (signal: AbortSignal) => {
      if (!filters || !hasFilters) return createListing(signal);
      const metadata = new Map<
        string,
        Promise<Record<string, unknown> | null>
      >();
      const requestJson = async (path: string) => {
        const cached = filterMetadata.get(path);
        if (cached && cached.expires > Date.now()) return cached.data;
        const response = await fetchWithTimeout(
          `${hfEndpoint}${path}`,
          {
            signal,
            ...(accessToken
              ? { headers: { Authorization: `Bearer ${accessToken}` } }
              : {}),
          },
          HF_SEARCH_TIMEOUT_MS,
        );
        if ([401, 403].includes(response.status)) return null;
        if (!response.ok && response.status !== 404) {
          throw new Error(
            `Model filter metadata request failed (HTTP ${response.status})`,
          );
        }
        try {
          const data = response.status === 404 ? null : await response.json();
          if (!signal.aborted) {
            filterMetadata.set(path, {
              data,
              expires: Date.now() + FILTER_METADATA_TTL_MS,
            });
          }
          return data;
        } catch {
          return null;
        }
      };
      const fetchJson = (path: string) => {
        let request = metadata.get(path);
        if (!request) {
          request = requestJson(path);
          metadata.set(path, request);
        }
        return request;
      };
      const filterPriority: ModelListingTransform = (iterator) =>
        filterModelListing(iterator, filters, fetchJson, undefined, keepRow);
      const iterator = createListing(signal, filterPriority);
      const pinnedPromise =
        pinnedId &&
        !unslothOnly &&
        !channelOwner &&
        !channelTagsKey &&
        !channelQuery
          ? modelInfo({
              hubUrl: hfEndpoint,
              name: pinnedId,
              additionalFields: ALL_FIELDS,
              fetch: makeHfFetch(signal),
              ...(accessToken ? { credentials: { accessToken } } : {}),
            }).catch(() => null)
          : undefined;
      return filterModelListing(
        iterator,
        filters,
        fetchJson,
        pinnedPromise,
        keepRow,
      );
    },
    [
      createListing,
      filters,
      hasFilters,
      pinnedId,
      unslothOnly,
      channelOwner,
      channelTagsKey,
      channelQuery,
      hfEndpoint,
      accessToken,
      filterMetadata,
      keepRow,
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
