// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  PersistStorage,
  StateStorage,
  StorageValue,
} from "zustand/middleware";

type DebouncedPersistStorageOptions = {
  delayMs?: number;
  getStorage?: () => StateStorage<void>;
};

export type DebouncedPersistStorage<S> = PersistStorage<S, void> & {
  flush: () => void;
};

function parseStoredValue<S>(value: string | null): StorageValue<S> | null {
  if (value === null) return null;
  return JSON.parse(value) as StorageValue<S>;
}

export function createDebouncedJSONPersistStorage<S>({
  delayMs = 250,
  getStorage = () => window.localStorage,
}: DebouncedPersistStorageOptions = {}): DebouncedPersistStorage<S> | undefined {
  let storage: StateStorage<void>;
  try {
    storage = getStorage();
  } catch {
    return undefined;
  }

  let pendingName: string | null = null;
  let pendingValue: StorageValue<S> | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const clearTimer = () => {
    if (timer === null) return;
    clearTimeout(timer);
    timer = null;
  };

  const flush = () => {
    if (pendingName === null || pendingValue === null) return;
    const name = pendingName;
    const value = pendingValue;
    pendingName = null;
    pendingValue = null;
    clearTimer();
    storage.setItem(name, JSON.stringify(value));
  };

  const scheduleFlush = () => {
    clearTimer();
    if (delayMs <= 0) {
      flush();
      return;
    }
    timer = setTimeout(flush, delayMs);
  };

  if (typeof window !== "undefined") {
    window.addEventListener("pagehide", flush);
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") flush();
    });
  }

  return {
    getItem: (name) => {
      if (pendingName === name && pendingValue !== null) {
        return pendingValue;
      }
      const stored = storage.getItem(name);
      if (stored instanceof Promise) {
        return stored.then(parseStoredValue<S>);
      }
      return parseStoredValue<S>(stored);
    },
    setItem: (name, value) => {
      pendingName = name;
      pendingValue = value;
      scheduleFlush();
    },
    removeItem: (name) => {
      if (pendingName === name) {
        pendingName = null;
        pendingValue = null;
        clearTimer();
      }
      storage.removeItem(name);
    },
    flush,
  };
}
