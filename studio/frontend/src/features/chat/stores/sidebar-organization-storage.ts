// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { strFromU8, strToU8, unzlibSync, zlibSync } from "fflate";
import type { StateStorage } from "zustand/middleware";

const COMPRESSED_PREFIX = "sidebar-zlib-v1:";
let lastStoredValue: string | null = null;

export function sidebarOrganizationStorageChanged(key: string): boolean {
  return window.localStorage.getItem(key) !== lastStoredValue;
}

export function unpackSidebarOrganization(value: string | null): string | null {
  if (!value?.startsWith(COMPRESSED_PREFIX)) return value;
  const bytes = strToU8(atob(value.slice(COMPRESSED_PREFIX.length)), true);
  return strFromU8(unzlibSync(bytes));
}

export function sidebarOrganizationStorage(): StateStorage {
  const storage = window.localStorage;
  function write(key: string, value: string): void {
    storage.setItem(key, value);
    lastStoredValue = value;
  }
  return {
    getItem(key) {
      lastStoredValue = storage.getItem(key);
      return unpackSidebarOrganization(lastStoredValue);
    },
    setItem(key, value) {
      try {
        write(key, value);
      } catch (error) {
        if (!(error instanceof DOMException) || error.name !== "QuotaExceededError") throw error;
        // Keep the visible layout and its sync journal in one atomic write for peer tabs.
        const compressed = zlibSync(strToU8(value), { level: 1 });
        const encoded = COMPRESSED_PREFIX + btoa(strFromU8(compressed, true));
        if (encoded.length >= value.length) throw error;
        write(key, encoded);
      }
    },
    removeItem(key) {
      storage.removeItem(key);
      lastStoredValue = null;
    },
  };
}
