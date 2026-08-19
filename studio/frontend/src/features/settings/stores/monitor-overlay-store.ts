// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface MonitorOverlayState {
  isOpen: boolean;
  isMinimized: boolean;
  setIsOpen: (open: boolean) => void;
  toggleMinimized: () => void;
}

export const useMonitorOverlayStore = create<MonitorOverlayState>()(
  persist(
    (set) => ({
      isOpen: false,
      isMinimized: false,
      setIsOpen: (isOpen) => set({ isOpen }),
      toggleMinimized: () => set((state) => ({ isMinimized: !state.isMinimized })),
    }),
    { name: "unsloth_monitor_overlay" }
  )
);