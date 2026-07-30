// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type InfiniteScrollResetKey = string | number | boolean | null;

export interface InfiniteScrollProgressInput {
  wasEnabled: boolean;
  signal: number;
  previousSignal: number;
  resultCount: number;
  previousResultCount: number;
  resetKey: InfiniteScrollResetKey;
  previousResetKey: InfiniteScrollResetKey;
}

export type InfiniteScrollProgress = "none" | "reset" | "visible-results";

export function resolveInfiniteScrollProgress({
  wasEnabled,
  signal,
  previousSignal,
  resultCount,
  previousResultCount,
  resetKey,
  previousResetKey,
}: InfiniteScrollProgressInput): InfiniteScrollProgress {
  if (
    !wasEnabled ||
    signal < previousSignal ||
    resultCount < previousResultCount ||
    resetKey !== previousResetKey
  ) {
    return "reset";
  }
  return resultCount > previousResultCount ? "visible-results" : "none";
}

export interface AutomaticFetchPolicyInput {
  enabled: boolean;
  isFetching: boolean;
  manualFetchAvailable: boolean;
  signal: number;
  lastRequestedSignal: number | null;
  autoFireCount: number;
  maxAutoFillFetches: number;
  manualFetchAfterAutoFill: boolean;
  hasScrollableOverflow: boolean;
  sentinelWithinPrefetchRange: boolean;
}

export type AutomaticFetchAction = "none" | "request" | "offer-manual";

export function resolveAutomaticFetchAction({
  enabled,
  isFetching,
  manualFetchAvailable,
  signal,
  lastRequestedSignal,
  autoFireCount,
  maxAutoFillFetches,
  manualFetchAfterAutoFill,
  hasScrollableOverflow,
  sentinelWithinPrefetchRange,
}: AutomaticFetchPolicyInput): AutomaticFetchAction {
  if (!enabled || isFetching || manualFetchAvailable) {
    return "none";
  }
  if (lastRequestedSignal !== null && signal <= lastRequestedSignal) {
    return "none";
  }
  if (hasScrollableOverflow && !sentinelWithinPrefetchRange) {
    return "none";
  }
  if (autoFireCount >= maxAutoFillFetches) {
    return manualFetchAfterAutoFill ? "offer-manual" : "none";
  }
  return "request";
}
