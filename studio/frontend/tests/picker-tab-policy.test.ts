// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  resolveInferredPickerTabLock,
  resolvePickerDeviceListState,
  resolvePickerTab,
  shouldRefreshPickerInventoryOnMount,
} from "../src/components/resource-picker/picker-tab-policy.ts";

const inferredBase = {
  hasExplicitTabPreference: false,
  isDeviceInventorySettled: false,
  online: true,
  selectedTab: "hub" as const,
};

test("locks a cold inferred picker to its opening tab", () => {
  const loading = {
    ...inferredBase,
    hasDeviceItems: false,
  };
  assert.equal(
    resolvePickerTab({ ...loading, lockedInferredTab: null }),
    "hub",
  );
  assert.equal(resolveInferredPickerTabLock(loading), "hub");

  const withUnsettledDeviceItems = {
    ...loading,
    hasDeviceItems: true,
  };
  assert.equal(
    resolvePickerTab({
      ...withUnsettledDeviceItems,
      lockedInferredTab: null,
    }),
    "hub",
  );
  assert.equal(resolveInferredPickerTabLock(withUnsettledDeviceItems), "hub");

  const withDeviceItems = {
    ...withUnsettledDeviceItems,
    isDeviceInventorySettled: true,
  };
  assert.equal(
    resolvePickerTab({ ...withDeviceItems, lockedInferredTab: null }),
    "device",
  );
  assert.equal(resolveInferredPickerTabLock(withDeviceItems), "device");

  const withoutDeviceItems = {
    ...loading,
    isDeviceInventorySettled: true,
  };
  assert.equal(
    resolvePickerTab({ ...withoutDeviceItems, lockedInferredTab: null }),
    "hub",
  );
  assert.equal(resolveInferredPickerTabLock(withoutDeviceItems), "hub");
});

test("keeps partial-failure retry available when filtering removes every row", () => {
  assert.equal(
    resolvePickerDeviceListState({
      error: null,
      hasItems: false,
      hasQuery: true,
      isLoading: false,
      warning: true,
    }),
    "warning",
  );
  assert.equal(
    resolvePickerDeviceListState({
      error: null,
      hasItems: false,
      hasQuery: true,
      isLoading: false,
      warning: false,
    }),
    "no-results",
  );
  assert.equal(
    resolvePickerDeviceListState({
      error: null,
      hasItems: true,
      hasQuery: true,
      isLoading: false,
      warning: true,
    }),
    "items",
  );
  assert.equal(
    resolvePickerDeviceListState({
      error: null,
      hasItems: false,
      hasQuery: true,
      isLoading: true,
      warning: true,
    }),
    "loading",
  );
});

test("keeps the opening tab stable when inventory settles", () => {
  const loading = {
    ...inferredBase,
    hasDeviceItems: false,
  };
  const lock = resolveInferredPickerTabLock(loading);
  assert.equal(lock, "hub");
  assert.equal(
    resolvePickerTab({
      ...loading,
      hasDeviceItems: true,
      isDeviceInventorySettled: true,
      lockedInferredTab: lock,
    }),
    "hub",
  );
});

test("refreshes picker inventory only when it was ready at mount", () => {
  assert.deepEqual([false, true].map(shouldRefreshPickerInventoryOnMount), [false, true]);
});

test("preserves explicit, locked, and offline tab decisions", () => {
  assert.equal(
    resolvePickerTab({
      ...inferredBase,
      hasDeviceItems: true,
      hasExplicitTabPreference: true,
      lockedInferredTab: null,
      selectedTab: "hub",
    }),
    "hub",
  );
  assert.equal(
    resolveInferredPickerTabLock({
      ...inferredBase,
      hasDeviceItems: true,
      hasExplicitTabPreference: true,
    }),
    null,
  );
  assert.equal(
    resolvePickerTab({
      ...inferredBase,
      hasDeviceItems: true,
      isDeviceInventorySettled: true,
      lockedInferredTab: "hub",
    }),
    "hub",
  );
  assert.equal(
    resolvePickerTab({
      ...inferredBase,
      hasDeviceItems: false,
      lockedInferredTab: null,
      online: false,
    }),
    "device",
  );
});

test("an empty settled inventory overrides a saved On Device preference", () => {
  assert.equal(
    resolvePickerTab({
      ...inferredBase,
      hasDeviceItems: false,
      isDeviceInventorySettled: true,
      hasExplicitTabPreference: true,
      selectedTab: "device",
      lockedInferredTab: null,
    }),
    "hub",
  );
});

test("an empty inventory replaces a stale inferred Device tab after scanning", () => {
  assert.equal(
    resolvePickerTab({
      ...inferredBase,
      hasDeviceItems: false,
      isDeviceInventorySettled: true,
      lockedInferredTab: "device",
    }),
    "hub",
  );
});

test("a current visit's deliberate Device choice remains available when empty", () => {
  const input = {
    ...inferredBase,
    hasDeviceItems: false,
    isDeviceInventorySettled: true,
    hasExplicitTabPreference: true,
    hasCurrentTabChoice: true,
    selectedTab: "device" as const,
    lockedInferredTab: null,
  };
  assert.equal(resolvePickerTab(input), "device");
  // Closing clears the current choice; reopening defaults to recommendations again.
  assert.equal(resolvePickerTab({ ...input, hasCurrentTabChoice: false }), "hub");
  assert.equal(resolvePickerTab({ ...input, hasDeviceItems: true }), "device");
});
