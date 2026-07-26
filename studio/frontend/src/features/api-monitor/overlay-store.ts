// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

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
      partialize: (state) => ({ autoOpen: state.autoOpen }),
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
