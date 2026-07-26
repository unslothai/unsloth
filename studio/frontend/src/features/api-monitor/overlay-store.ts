// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

/**
 * localStorage that cannot throw.
 *
 * Safari private browsing, Firefox with the origin's cookies blocked, and an
 * opaque origin in a webview all make `window.localStorage` throw on access
 * rather than return null. Losing the preference there is fine; taking the
 * whole panel down with it is not.
 */
const safeStorage = {
  getItem: (name: string): string | null => {
    try {
      return window.localStorage.getItem(name);
    } catch {
      return null;
    }
  },
  setItem: (name: string, value: string): void => {
    try {
      window.localStorage.setItem(name, value);
    } catch {
      // Quota exceeded or storage denied. The preference stays session-only.
    }
  },
  removeItem: (name: string): void => {
    try {
      window.localStorage.removeItem(name);
    } catch {
      // Same.
    }
  },
};

interface ApiMonitorOverlayState {
  /** Whether the floating panel is on screen right now. Session state. */
  isOpen: boolean;
  /** Set on close so the panel does not pop back during the same burst. */
  suppressed: boolean;
  /** Persisted opt out: when false the panel never opens itself. */
  autoOpen: boolean;
  open: () => void;
  close: () => void;
  setAutoOpen: (autoOpen: boolean) => void;
}

/**
 * Only `autoOpen` persists. Open/closed is session state: a dismissal lasts the
 * sitting, not forever.
 */
export const useApiMonitorOverlayStore = create<ApiMonitorOverlayState>()(
  persist(
    (set) => ({
      isOpen: false,
      suppressed: false,
      autoOpen: true,
      open: () => set({ isOpen: true, suppressed: false }),
      close: () => set({ isOpen: false, suppressed: true }),
      setAutoOpen: (autoOpen) => set({ autoOpen }),
    }),
    {
      name: "unsloth_api_monitor_overlay",
      version: 1,
      storage: createJSONStorage(() => safeStorage),
      partialize: (state) => ({ autoOpen: state.autoOpen }),
      // Without this a version bump discards the payload, quietly handing the
      // popup back to someone who had turned it off.
      migrate: (persisted) => persisted,
      // Explicit merge so an older stored payload cannot resurrect `isOpen`.
      merge: (persisted, current) => ({
        ...current,
        autoOpen:
          typeof (persisted as { autoOpen?: unknown } | null)?.autoOpen ===
          "boolean"
            ? (persisted as { autoOpen: boolean }).autoOpen
            : current.autoOpen,
      }),
    },
  ),
);
