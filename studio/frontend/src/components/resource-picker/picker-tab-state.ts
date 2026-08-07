// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const PICKER_TAB = {
  device: "device",
  hub: "hub",
} as const;

export const PICKER_TAB_VALUES = [PICKER_TAB.device, PICKER_TAB.hub] as const;
export type PickerTab = (typeof PICKER_TAB_VALUES)[number];

function isPickerTab(value: unknown): value is PickerTab {
  return (
    typeof value === "string" &&
    (PICKER_TAB_VALUES as readonly string[]).includes(value)
  );
}

export function readPickerTabPreference(storageKey: string): PickerTab | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const value = window.localStorage.getItem(storageKey);
    return isPickerTab(value) ? value : null;
  } catch {
    return null;
  }
}

export function writePickerTabPreference(
  storageKey: string,
  tab: PickerTab,
): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(storageKey, tab);
  } catch {
    return;
  }
}

export function pickerTabId(idBase: string, value: string): string {
  return `${idBase}-tab-${value}`;
}
