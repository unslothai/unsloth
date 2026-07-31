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

test("keeps a cold picker inferable until device inventory settles", () => {
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
    "device",
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
  assert.equal(resolveInferredPickerTabLock(withoutDeviceItems), "hub");
});

test("preserves explicit, locked, and offline tab decisions", () => {
  assert.equal(
    resolvePickerTab({
      ...inferredBase,
      hasDeviceItems: true,
      hasExplicitTabPreference: true,
      isDeviceInventorySettled: true,
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
      isDeviceInventorySettled: true,
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
