// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback } from "react";

interface PickerHubPaginationParams {
  enabled: boolean;
  fetchMore: () => boolean;
  hasMore: boolean;
  isFetching: boolean;
  resetKey: string | number | boolean | null;
  resultCount: number;
  scannedCount: number;
}

export function usePickerHubPagination({
  enabled,
  fetchMore,
  hasMore,
  isFetching,
  resetKey,
  resultCount,
  scannedCount,
}: PickerHubPaginationParams) {
  const canFetch = enabled && hasMore;
  const fetchMoreIfAvailable = useCallback(() => {
    if (!canFetch) {
      return false;
    }
    return fetchMore();
  }, [canFetch, fetchMore]);

  return {
    fetchMore: fetchMoreIfAvailable,
    signal: scannedCount,
    options: {
      enabled: canFetch,
      isFetching,
      manualFetchAfterAutoFill: true,
      maxAutoFillFetches: 5,
      resetKey,
      resultCount,
    },
  };
}
