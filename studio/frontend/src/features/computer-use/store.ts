// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { syncAcrossWindows } from "@/lib/sync-across-windows";
import {
  type ComputerUseStatus,
  allowComputerUseApp,
  loadComputerUseStatus,
  resumeComputerUse,
  revokeComputerUse,
} from "./api";

type ComputerUseState = {
  /** The composer's Computer use switch: offer the model the computer tool. */
  enabled: boolean;
  setEnabled: (enabled: boolean) => void;
  /** What the server last said; null until asked, or on a backend without computer use. */
  status: ComputerUseStatus | null;
  setStatus: (status: ComputerUseStatus | null) => void;
};

export const useComputerUseStore = create<ComputerUseState>()(
  persist(
    (set, get) => ({
      enabled: false,
      setEnabled: (enabled) => {
        // Switching it off also withdraws the go-ahead every chat was given.
        if (!enabled && get().enabled) void revokeComputerUse().catch(() => undefined);
        // Switching it back on lifts that, for a chat with Full access that has no card to ask on.
        if (enabled && !get().enabled) {
          void resumeComputerUse()
            .then((status) => set({ status }))
            .catch(() => undefined);
        }
        set({ enabled });
      },
      status: null,
      setStatus: (status) => set({ status }),
    }),
    {
      name: "unsloth_computer_use",
      partialize: (state) => ({ enabled: state.enabled }),
      // Storage another build wrote must never read as on.
      merge: (persisted, current) => ({
        ...current,
        enabled: (persisted as { enabled?: unknown } | null)?.enabled === true,
      }),
    },
  ),
);
syncAcrossWindows(useComputerUseStore);

let inFlight: Promise<ComputerUseStatus | null> | null = null;

/** One request at a time; a failed read leaves the last answer in place. */
export function refreshComputerUseStatus(): Promise<ComputerUseStatus | null> {
  inFlight ??= loadComputerUseStatus()
    .then((status) => {
      useComputerUseStore.getState().setStatus(status);
      return status;
    })
    .catch(() => useComputerUseStore.getState().status)
    .finally(() => {
      inFlight = null;
    });
  return inFlight;
}

/** The approval card's Always allow: adds the app to the list in Settings > Computer use. */
export async function alwaysAllowComputerUseApp(name: string): Promise<void> {
  useComputerUseStore.getState().setStatus(await allowComputerUseApp(name));
}

/** The server would hand a chat the tool: it can control this computer and the owner switched it on. */
export function computerUseReady(status: ComputerUseStatus | null): boolean {
  return !!status && status.available && status.enabled;
}
