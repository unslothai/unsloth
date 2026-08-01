// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PICKER_TAB, type PickerTab } from "./picker-tab-state.ts";

export interface PickerDeviceInventoryState {
  hasDeviceItems: boolean;
  isDeviceInventorySettled: boolean;
}

export interface PickerTabResolutionInput extends PickerDeviceInventoryState {
  hasExplicitTabPreference: boolean;
  lockedInferredTab: PickerTab | null;
  online: boolean;
  selectedTab: PickerTab;
}

export function resolvePickerTab({
  hasDeviceItems,
  hasExplicitTabPreference,
  isDeviceInventorySettled,
  lockedInferredTab,
  online,
  selectedTab,
}: PickerTabResolutionInput): PickerTab {
  const shouldUseHubTab = online && isDeviceInventorySettled && !hasDeviceItems;
  const inferredTab = hasExplicitTabPreference
    ? selectedTab
    : isDeviceInventorySettled
      ? shouldUseHubTab
        ? PICKER_TAB.hub
        : PICKER_TAB.device
      : online
        ? PICKER_TAB.hub
        : PICKER_TAB.device;
  return lockedInferredTab ?? inferredTab;
}

export function resolveInferredPickerTabLock(
  input: Omit<PickerTabResolutionInput, "lockedInferredTab">,
): PickerTab | null {
  if (input.hasExplicitTabPreference) {
    return null;
  }
  return resolvePickerTab({ ...input, lockedInferredTab: null });
}
