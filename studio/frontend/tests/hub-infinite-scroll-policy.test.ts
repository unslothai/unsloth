// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type AutomaticFetchPolicyInput,
  type InfiniteScrollProgressInput,
  resolveAutomaticFetchAction,
  resolveInfiniteScrollProgress,
} from "../src/features/hub/hooks/hub-infinite-scroll-policy.ts";

const automaticBase: AutomaticFetchPolicyInput = {
  enabled: true,
  isFetching: false,
  manualFetchAvailable: false,
  signal: 96,
  lastRequestedSignal: 48,
  autoFireCount: 0,
  maxAutoFillFetches: 5,
  manualFetchAfterAutoFill: true,
  hasScrollableOverflow: true,
  sentinelWithinPrefetchRange: true,
};

function automatic(overrides: Partial<AutomaticFetchPolicyInput> = {}) {
  return resolveAutomaticFetchAction({ ...automaticBase, ...overrides });
}

test("automatic paging waits for progress after an accepted request", () => {
  assert.equal(automatic({ signal: 48, lastRequestedSignal: 48 }), "none");
  assert.equal(
    automatic({
      signal: 48,
      lastRequestedSignal: 48,
      hasScrollableOverflow: false,
      sentinelWithinPrefetchRange: false,
    }),
    "none",
  );
  assert.equal(automatic({ signal: 47, lastRequestedSignal: 48 }), "none");
  assert.equal(automatic({ signal: 49, lastRequestedSignal: 48 }), "request");
  assert.equal(automatic({ lastRequestedSignal: null }), "request");
});

test("automatic paging stays idle while unavailable or already fetching", () => {
  assert.equal(automatic({ enabled: false }), "none");
  assert.equal(automatic({ isFetching: true }), "none");
  assert.equal(automatic({ manualFetchAvailable: true }), "none");
});

test("the automatic cap enters only the configured manual fallback", () => {
  assert.equal(
    automatic({ autoFireCount: 5, maxAutoFillFetches: 5 }),
    "offer-manual",
  );
  assert.equal(
    automatic({
      autoFireCount: 5,
      maxAutoFillFetches: 5,
      manualFetchAfterAutoFill: false,
    }),
    "none",
  );
  assert.equal(
    automatic({ autoFireCount: 4, maxAutoFillFetches: 5 }),
    "request",
  );
});

test("prefetch geometry gates only an overflowing container", () => {
  assert.equal(
    automatic({
      hasScrollableOverflow: true,
      sentinelWithinPrefetchRange: false,
    }),
    "none",
  );
  assert.equal(
    automatic({
      hasScrollableOverflow: true,
      sentinelWithinPrefetchRange: true,
    }),
    "request",
  );
  assert.equal(
    automatic({
      hasScrollableOverflow: false,
      sentinelWithinPrefetchRange: false,
    }),
    "request",
  );
});

const progressBase: InfiniteScrollProgressInput = {
  wasEnabled: true,
  signal: 96,
  previousSignal: 48,
  resultCount: 24,
  previousResultCount: 24,
  resetKey: "query",
  previousResetKey: "query",
};

function progress(overrides: Partial<InfiniteScrollProgressInput> = {}) {
  return resolveInfiniteScrollProgress({ ...progressBase, ...overrides });
}

test("listing identity and backwards progress reset automatic paging", () => {
  assert.equal(progress({ wasEnabled: false }), "reset");
  assert.equal(progress({ signal: 47, previousSignal: 48 }), "reset");
  assert.equal(progress({ resultCount: 23, previousResultCount: 24 }), "reset");
  assert.equal(progress({ resetKey: "next-query" }), "reset");
});

test("visible growth resets the cap without treating raw progress as a reset", () => {
  assert.equal(
    progress({ resultCount: 25, previousResultCount: 24 }),
    "visible-results",
  );
  assert.equal(progress(), "none");
});
