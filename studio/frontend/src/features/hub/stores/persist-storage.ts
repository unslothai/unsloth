// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { StateStorage } from "zustand/middleware";

export const noopStorage: StateStorage = {
  getItem: () => null,
  setItem: () => undefined,
  removeItem: () => undefined,
};

export function createThrottledStorage(
  base: StateStorage,
  delayMs: number,
): StateStorage {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending: { name: string; value: string } | null = null;
  const flush = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    if (pending) {
      base.setItem(pending.name, pending.value);
      pending = null;
    }
  };
  if (typeof window !== "undefined") {
    window.addEventListener("pagehide", flush);
  }
  return {
    getItem: (name) => base.getItem(name),
    setItem: (name, value) => {
      pending = { name, value };
      if (timer === null) timer = setTimeout(flush, delayMs);
    },
    removeItem: (name) => {
      pending = null;
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      base.removeItem(name);
    },
  };
}
