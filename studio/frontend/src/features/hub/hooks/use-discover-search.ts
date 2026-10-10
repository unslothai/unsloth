// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef } from "react";
import { useHubName } from "@/lib/hf-endpoint";
import { toast } from "@/lib/toast";
import {
  clearRemoteBackoff,
  hubAuthFailure,
  type HubFailure,
} from "../lib/network";
import { useHubAvailability } from "./use-online-status";
import {
  type HfModelResult,
  type HfModelSearchChannel,
  type HfSortDirection,
  type HfSortKey,
  useHubModelSearch,
} from "./use-hub-model-search";
import {
  type HfDatasetResult,
  useHubDatasetSearch,
} from "./use-hub-dataset-search";

export interface DiscoverSearch {
  results: HfModelResult[];
  datasetResults: HfDatasetResult[];
  scannedCount: number;
  isLoading: boolean;
  isLoadingMore: boolean;
  hasMore: boolean;
  fetchMore: () => boolean;
  searchError: string | null;
  searchFailure: HubFailure | null;
  handleRetrySearch: () => void;
}

type DiscoverErrorKind =
  | "offline"
  | "auth"
  | "rate-limited"
  | "server"
  | "unknown";

const RECONNECT_RETRY_COOLDOWN_MS = 90_000;

function classifyDiscoverError(
  message: string,
  online: boolean,
): DiscoverErrorKind {
  if (!online) return "offline";
  const lower = message.toLowerCase();
  if (
    lower.includes("429") ||
    lower.includes("rate limit") ||
    lower.includes("too many requests")
  ) {
    return "rate-limited";
  }
  if (
    lower.includes("401") ||
    lower.includes("403") ||
    lower.includes("unauthorized") ||
    lower.includes("forbidden") ||
    (lower.includes("token") && !lower.includes("unexpected token")) ||
    lower.includes("authentication")
  ) {
    return "auth";
  }
  if (
    lower.includes("500") ||
    lower.includes("502") ||
    lower.includes("503") ||
    lower.includes("504") ||
    lower.includes("server")
  ) {
    return "server";
  }
  return "unknown";
}

function discoverErrorTitle(kind: DiscoverErrorKind, hub: string): string {
  switch (kind) {
    case "offline":
      return `Can't reach ${hub}`;
    case "auth":
      return `${hub} auth failed`;
    case "rate-limited":
      return `${hub} rate limit`;
    default:
      return `Couldn't reach ${hub}`;
  }
}

export function useDiscoverSearch({
  debouncedQuery,
  accessToken,
  isDiscoverTab,
  isDatasetMode,
  sortBy,
  direction,
  channel,
  ownerScope,
}: {
  debouncedQuery: string;
  accessToken: string | undefined;
  isDiscoverTab: boolean;
  isDatasetMode: boolean;
  sortBy: HfSortKey;
  direction: HfSortDirection;
  channel: HfModelSearchChannel | null;
  ownerScope: "unsloth" | "all";
}): DiscoverSearch {
  const { phase, failure } = useHubAvailability();
  const hub = useHubName();
  // "probing" counts: a lapsed backoff is when the next request should test the network.
  const canProbe = phase !== "unavailable";
  // Only a success promotes to "available", never a lapsed backoff.
  const online = phase === "available";

  // Gated on the live backoff only, or typing through an outage re-arms the window each tick.
  const modelSearch = useHubModelSearch(debouncedQuery, {
    accessToken,
    sortBy,
    sortDirection: direction,
    pinUnslothFirst: true,
    ownerScope,
    enabled: canProbe && isDiscoverTab && !isDatasetMode,
    keepUnsupportedTags: true,
    channel,
  });
  const datasetSearch = useHubDatasetSearch(debouncedQuery, {
    accessToken,
    // Not folded into `enabled`, which returns [] when false and would blank rendered rows.
    enabled: isDiscoverTab && isDatasetMode,
    paused: !canProbe,
    sortBy,
    sortDirection: direction,
  });

  const results = isDatasetMode ? [] : modelSearch.results;
  const isLoading = isDatasetMode ? datasetSearch.isLoading : modelSearch.isLoading;
  const isLoadingMore = isDatasetMode
    ? datasetSearch.isLoadingMore
    : modelSearch.isLoadingMore;
  const hasMore = isDatasetMode ? datasetSearch.hasMore : modelSearch.hasMore;
  const scannedCount = isDatasetMode
    ? datasetSearch.scannedCount
    : modelSearch.scannedCount;
  const rawFetchMore = isDatasetMode
    ? datasetSearch.fetchMore
    : modelSearch.fetchMore;
  const rawSearchError = isDatasetMode ? datasetSearch.error : modelSearch.error;
  const retrySearch = isDatasetMode ? datasetSearch.retry : modelSearch.retry;
  const needsRestart = isDatasetMode
    ? datasetSearch.needsRestart
    : modelSearch.needsRestart;
  const searchError = isDiscoverTab ? rawSearchError : null;
  // A 401 is not a network failure. Memoised: the toast effect depends on it.
  const searchFailure = useMemo(
    () =>
      isDiscoverTab
        ? (failure ?? hubAuthFailure({ message: rawSearchError }))
        : null,
    [isDiscoverTab, failure, rawSearchError],
  );
  const fetchMore = useCallback(() => {
    if (!canProbe || !hasMore) return false;
    // A failed page took the iterator with it; resuming would silently end pagination.
    if (needsRestart()) {
      retrySearch();
      return true;
    }
    return rawFetchMore();
  }, [canProbe, hasMore, needsRestart, rawFetchMore, retrySearch]);

  const handleRetrySearch = useCallback(() => {
    // Always re-probe so users can test a network fix without waiting out the timer.
    clearRemoteBackoff();
    retrySearch();
    toast.message("Retrying…", {
      description: "Reaching Hugging Face for the latest models.",
    });
  }, [retrySearch]);

  const lastErrorRef = useRef<DiscoverErrorKind | null>(null);
  useEffect(() => {
    if (!isDiscoverTab) {
      lastErrorRef.current = null;
      return;
    }
    if (!searchError) {
      lastErrorRef.current = null;
      return;
    }
    const errorKind = classifyDiscoverError(searchError, online);
    if (lastErrorRef.current === errorKind) return;
    lastErrorRef.current = errorKind;
    toast.error(discoverErrorTitle(errorKind, hub), {
      description: searchFailure?.message ?? searchError,
      action: { label: "Retry", onClick: handleRetrySearch },
    });
  }, [isDiscoverTab, searchError, searchFailure, online, handleRetrySearch, hub]);

  // Driven by a successful request, never a lapsed timer (that caused an offline/online loop).
  const wasUnavailableRef = useRef(phase !== "available");
  const lastReconnectAtRef = useRef(0);
  // Latched while this feed has a request in flight so the reconnect effect knows whose recovery it is.
  const selfProbedRef = useRef(false);
  useEffect(() => {
    if (isLoading || isLoadingMore) {
      selfProbedRef.current = true;
    } else if (!online) {
      selfProbedRef.current = false;
    }
  }, [online, isLoading, isLoadingMore]);
  useEffect(() => {
    if (online && wasUnavailableRef.current && isDiscoverTab) {
      const now = Date.now();
      if (now - lastReconnectAtRef.current > RECONNECT_RETRY_COOLDOWN_MS) {
        lastReconnectAtRef.current = now;
        const selfProbed = selfProbedRef.current;
        selfProbedRef.current = false;
        toast.success("Back online", {
          description: "Refreshing the discovery feed.",
        });
        // Only when another surface proved reachability; our own success already rendered results.
        if (!selfProbed) retrySearch();
      }
    }
    wasUnavailableRef.current = !online;
  }, [online, retrySearch, isDiscoverTab]);

  return {
    results,
    datasetResults: datasetSearch.results,
    scannedCount,
    isLoading,
    isLoadingMore,
    hasMore,
    fetchMore,
    searchError,
    searchFailure,
    handleRetrySearch,
  };
}
