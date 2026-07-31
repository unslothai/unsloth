// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  resolveInferredPickerTabLock,
  resolvePickerTab,
} from "../src/components/resource-picker/picker-tab-policy.ts";

const inferredBase = {
  hasExplicitTabPreference: false,
  isDeviceInventorySettled: false,
  online: true,
  selectedTab: "hub" as const,
};

test("keeps a cold inferred picker on its initial tab while inventory settles", () => {
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

test("does not change an inferred tab after the picker opens", () => {
  assert.equal(
    resolvePickerTab({
      ...inferredBase,
      hasDeviceItems: true,
      isDeviceInventorySettled: true,
      lockedInferredTab: "hub",
    }),
    "hub",
  );
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
