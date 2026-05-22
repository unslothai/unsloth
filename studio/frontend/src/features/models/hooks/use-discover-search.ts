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
  isLoading: boolean;
  isLoadingMore: boolean;
  fetchMore: () => void;
  searchError: string | null;
  handleRetrySearch: () => void;
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
    enabled: isDiscoverTab && !isDatasetMode,
    keepUnsupportedTags: true,
    channel: channelOption,
  });
  const datasetSearch = useHfDatasetSearch(debouncedQuery, {
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
  const fetchMore = isDatasetMode ? datasetSearch.fetchMore : modelSearch.fetchMore;
  const searchError = isDatasetMode ? datasetSearch.error : modelSearch.error;
  const retrySearch = isDatasetMode ? datasetSearch.retry : modelSearch.retry;

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

  const lastErrorRef = useRef<string | null>(null);
  useEffect(() => {
    if (!searchError) {
      lastErrorRef.current = null;
      return;
    }
    if (lastErrorRef.current === searchError) return;
    lastErrorRef.current = searchError;
    toast.error(online ? "Couldn't reach Hugging Face" : "You're offline", {
      description: online
        ? searchError
        : "Reconnect to the internet to browse models.",
      action: { label: "Retry", onClick: handleRetrySearch },
    });
  }, [searchError, online, handleRetrySearch]);

  const wasOfflineRef = useRef(!online);
  useEffect(() => {
    if (online && wasOfflineRef.current) {
      toast.success("Back online", {
        description: "Refreshing the discovery feed.",
      });
      retrySearch();
    }
    wasOfflineRef.current = !online;
  }, [online, retrySearch]);

  return {
    results,
    datasetResults: datasetSearch.results,
    isLoading,
    isLoadingMore,
    fetchMore,
    searchError,
    handleRetrySearch,
  };
}
