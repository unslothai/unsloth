// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef } from "react";
import { toast } from "sonner";
import {
  type HfDatasetResult,
  type HfModelResult,
  type HfSortDirection,
  type HfSortKey,
  useHfDatasetSearch,
  useHfModelSearch,
} from "@/hooks";
import type { ChannelPreset } from "../lib/channels";

export interface DiscoverSearch {
  results: HfModelResult[];
  datasetResults: HfDatasetResult[];
  scannedCount: number;
  isLoading: boolean;
  isLoadingMore: boolean;
  hasMore: boolean;
  fetchMore: () => boolean;
  searchError: string | null;
  handleRetrySearch: () => void;
}

type DiscoverErrorKind = "offline" | "auth" | "rate-limited" | "server" | "unknown";

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
    lower.includes("token") ||
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
      return "You're offline";
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
  activeChannel,
  online,
}: {
  debouncedQuery: string;
  accessToken: string | undefined;
  isDiscoverTab: boolean;
  isDatasetMode: boolean;
  sortBy: HfSortKey;
  direction: HfSortDirection;
  activeChannel: ChannelPreset | null;
  online: boolean;
}): DiscoverSearch {
  const channelOption = useMemo(
    () =>
      activeChannel
        ? {
            owner: activeChannel.owner,
            tags: activeChannel.tags,
            query: activeChannel.query,
            idSuffix: activeChannel.idSuffix,
          }
        : null,
    [activeChannel],
  );

  const modelSearch = useHfModelSearch(debouncedQuery, {
    accessToken,
    sortBy,
    sortDirection: direction,
    pinUnslothFirst:
      sortBy === "trendingScore" && direction === "desc" && !activeChannel,
    enabled: online && isDiscoverTab && !isDatasetMode,
    keepUnsupportedTags: true,
    channel: channelOption,
  });
  const datasetSearch = useHfDatasetSearch(debouncedQuery, {
    accessToken,
    enabled: online && isDiscoverTab && isDatasetMode,
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
  const searchError = isDiscoverTab && online ? rawSearchError : null;
  const fetchMore = useCallback(() => {
    if (!online || !hasMore) return false;
    return rawFetchMore();
  }, [online, hasMore, rawFetchMore]);

  const handleRetrySearch = useCallback(() => {
    if (!online) {
      toast.error("You're offline", {
        description: "Reconnect to the internet to browse Hugging Face.",
      });
      return;
    }
    retrySearch();
    toast.message("Retrying…", {
      description: "Reaching Hugging Face for the latest models.",
    });
  }, [online, retrySearch]);

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
      description: online
        ? searchError
        : "Reconnect to the internet to browse models.",
      action: { label: "Retry", onClick: handleRetrySearch },
    });
  }, [isDiscoverTab, searchError, online, handleRetrySearch]);

  const wasOfflineRef = useRef(!online);
  const lastReconnectAtRef = useRef(0);
  useEffect(() => {
    if (online && wasOfflineRef.current && isDiscoverTab) {
      const now = Date.now();
      if (now - lastReconnectAtRef.current > RECONNECT_RETRY_COOLDOWN_MS) {
        lastReconnectAtRef.current = now;
        toast.success("Back online", {
          description: "Refreshing the discovery feed.",
        });
        retrySearch();
      }
    }
    wasOfflineRef.current = !online;
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
    handleRetrySearch,
  };
}
