// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { strFromU8, strToU8, unzlibSync, zlibSync } from "fflate";
import type { StateStorage } from "zustand/middleware";

import { BROWSER_ACCOUNT_KEY } from "../../../lib/browser-account-key.ts";
import { SIDEBAR_SYNC_BASELINE_KEY } from "./sidebar-organization-keys.ts";

const COMPRESSED_PREFIX = "sidebar-zlib-v1:";
let lastStoredValue: string | null = null;
let lastBaseline: { account: string; layout: string } | undefined;

export function sidebarOrganizationStorageChanged(key: string): boolean {
  return window.localStorage.getItem(key) !== lastStoredValue;
}

export function unpackSidebarOrganization(value: string | null): string | null {
  if (!value?.startsWith(COMPRESSED_PREFIX)) return value;
  const bytes = strToU8(atob(value.slice(COMPRESSED_PREFIX.length)), true);
  return strFromU8(unzlibSync(bytes));
}

function packSidebarOrganization(value: string): string {
  return COMPRESSED_PREFIX + btoa(strFromU8(zlibSync(strToU8(value), { level: 1 }), true));
}

function storageIsFull(error: unknown): boolean {
  return error instanceof DOMException && error.name === "QuotaExceededError";
}

export function rememberSidebarSyncBaseline(layout: string): void {
  const storage = window.localStorage;
  const account = storage.getItem(BROWSER_ACCOUNT_KEY);
  if (!account) return;
  lastBaseline = { account, layout };
  try {
    storage.setItem(SIDEBAR_SYNC_BASELINE_KEY, packSidebarOrganization(JSON.stringify(lastBaseline)));
  } catch (error) {
    if (!storageIsFull(error)) throw error;
    storage.removeItem(SIDEBAR_SYNC_BASELINE_KEY);
  }
}

export function readSidebarSyncBaseline(): string | undefined {
  if (typeof window === "undefined") return undefined;
  const storage = window.localStorage;
  const account = storage.getItem(BROWSER_ACCOUNT_KEY);
  if (!account) return undefined;
  try {
    const cached = unpackSidebarOrganization(storage.getItem(SIDEBAR_SYNC_BASELINE_KEY));
    if (cached) {
      const saved = JSON.parse(cached);
      if (saved?.account === account && typeof saved.layout === "string") return saved.layout;
    }
  } catch {
    // The active tab can still recover its own baseline if the cache cannot be read.
  }
  return lastBaseline?.account === account ? lastBaseline.layout : undefined;
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
        if (!storageIsFull(error)) throw error;
        if (storage.getItem(SIDEBAR_SYNC_BASELINE_KEY) !== null) {
          // The current layout and pending write take precedence over the recovery copy.
          storage.removeItem(SIDEBAR_SYNC_BASELINE_KEY);
          try {
            write(key, value);
            return;
          } catch (retryError) {
            if (!storageIsFull(retryError)) throw retryError;
          }
        }
        // Keep the visible layout and its sync journal in one atomic write for peer tabs.
        const encoded = packSidebarOrganization(value);
        if (encoded.length >= value.length) throw error;
        write(key, encoded);
      }
    },
    removeItem(key) {
      storage.removeItem(key);
      storage.removeItem(SIDEBAR_SYNC_BASELINE_KEY);
      lastBaseline = undefined;
      lastStoredValue = null;
    },
  };
}
