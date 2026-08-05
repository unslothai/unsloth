// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef } from "react";
import { toast } from "@/lib/toast";
import { clearRemoteBackoff, type HubFailure } from "@/features/hub/lib/network";
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
  fetchMoreManual: () => boolean;
  searchError: string | null;
  /** Classified cause of the last failure, for a diagnosable error panel. */
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

function discoverErrorTitle(kind: DiscoverErrorKind): string {
  switch (kind) {
    case "offline":
      return "Can't reach Hugging Face";
    case "auth":
      return "Hugging Face auth failed";
    case "rate-limited":
      return "Hugging Face rate limit";
    default:
      return "Couldn't reach Hugging Face";
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
  /** Accepted for compatibility; availability is read from the network store. */
  online?: boolean;
}): DiscoverSearch {
  const { phase, failure, proxyServing } = useHubAvailability();
  const online = phase === "available";

  // Not gated on availability: gating disabled the paginated hook, which
  // discarded the error, so every cause rendered as the same generic panel.
  const modelSearch = useHubModelSearch(debouncedQuery, {
    accessToken,
    sortBy,
    sortDirection: direction,
    pinUnslothFirst: true,
    ownerScope,
    enabled: isDiscoverTab && !isDatasetMode,
    keepUnsupportedTags: true,
    channel,
  });
  const datasetSearch = useHubDatasetSearch(debouncedQuery, {
    accessToken,
    enabled: isDiscoverTab && isDatasetMode,
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
  // Already sanitized in useHubPaginatedSearch, where every consumer reads it.
  const rawSearchError = isDatasetMode ? datasetSearch.error : modelSearch.error;
  const retrySearch = isDatasetMode ? datasetSearch.retry : modelSearch.retry;
  const needsRestart = isDatasetMode
    ? datasetSearch.needsRestart
    : modelSearch.needsRestart;
  // Surfaced regardless of availability: the failure IS the thing worth showing.
  const searchError = isDiscoverTab ? rawSearchError : null;
  const searchFailure = isDiscoverTab ? failure : null;
  // Allowed while probing: gating on `online` left Load more a permanent no-op.
  // The backoff still blocks the `unavailable` window.
  const canProbe = online || phase === "probing";
  const fetchMore = useCallback(() => {
    if (!canProbe || !hasMore) return false;
    return rawFetchMore();
  }, [canProbe, hasMore, rawFetchMore]);
  // The click path, on the same contract as Retry: the footer renders on
  // hasMore and so outlives the failed page, and a visible button that silently
  // does nothing for the whole backoff window is worse than a failed probe.
  const fetchMoreManual = useCallback(() => {
    if (!hasMore) return false;
    clearRemoteBackoff();
    // A page that failed took the iterator with it, so resuming would resolve
    // done and quietly end pagination. Restarting is the only way to continue.
    if (needsRestart()) {
      retrySearch();
      return true;
    }
    return rawFetchMore();
  }, [hasMore, needsRestart, rawFetchMore, retrySearch]);

  const handleRetrySearch = useCallback(() => {
    // Always re-probe: refusing during the backoff left users unable to test a
    // firewall, DNS or certificate change without waiting out the timer.
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
    toast.error(discoverErrorTitle(errorKind), {
      // The classified failure names the cause; the raw message covers HTTP
      // errors that never reach the network layer.
      description: searchFailure?.message ?? searchError,
      action: { label: "Retry", onClick: handleRetrySearch },
    });
  }, [isDiscoverTab, searchError, searchFailure, online, handleRetrySearch]);

  // Driven by a successful request, never a lapsed timer. Announcing recovery
  // on TTL expiry produced a permanent offline/back-online loop.
  const wasUnavailableRef = useRef(phase !== "available");
  const lastReconnectAtRef = useRef(0);
  useEffect(() => {
    // Not on proxy-served availability: retrySearch would build a fresh
    // transport with no affinity and re-attempt the blocked direct request.
    if (online && !proxyServing && wasUnavailableRef.current && isDiscoverTab) {
      const now = Date.now();
      if (now - lastReconnectAtRef.current > RECONNECT_RETRY_COOLDOWN_MS) {
        lastReconnectAtRef.current = now;
        toast.success("Back online", {
          description: "Refreshing the discovery feed.",
        });
        retrySearch();
      }
    }
    wasUnavailableRef.current = !online;
  }, [online, proxyServing, retrySearch, isDiscoverTab]);

  return {
    results,
    datasetResults: datasetSearch.results,
    scannedCount,
    isLoading,
    isLoadingMore,
    hasMore,
    fetchMore,
    fetchMoreManual,
    searchError,
    searchFailure,
    handleRetrySearch,
  };
}
