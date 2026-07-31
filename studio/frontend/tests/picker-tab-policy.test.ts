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

test("defers inferred tab locking until device inventory settles", () => {
  const loading = {
    ...inferredBase,
    hasDeviceItems: false,
  };
  assert.equal(
    resolvePickerTab({ ...loading, lockedInferredTab: null }),
    "hub",
  );
  assert.equal(resolveInferredPickerTabLock(loading), null);

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
  assert.equal(resolveInferredPickerTabLock(withUnsettledDeviceItems), null);

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

test("preserves a locked tab after user interaction", () => {
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

test("moves an idle cold picker to device when inventory arrives", () => {
  const loading = {
    ...inferredBase,
    hasDeviceItems: false,
  };
  const lock = resolveInferredPickerTabLock(loading);
  assert.equal(lock, null);
  assert.equal(
    resolvePickerTab({
      ...loading,
      hasDeviceItems: true,
      isDeviceInventorySettled: true,
      lockedInferredTab: lock,
    }),
    "device",
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
