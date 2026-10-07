// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Per account, like history: one account's "Always allow" must not skip another's prompt.

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { accountDatabaseName } from "@/lib/account-transition";

export type DownloadSiteDecision = "allow" | "block";

interface DownloadSitesState {
  sites: Record<string, DownloadSiteDecision>;
  setSite: (origin: string, decision: DownloadSiteDecision | null) => void;
}

export const useDownloadSitesStore = create<DownloadSitesState>()(
  persist(
    (set) => ({
      sites: {},
      setSite: (origin, decision) =>
        set((state) => {
          const sites = { ...state.sites };
          if (decision) sites[origin] = decision;
          else delete sites[origin];
          return { sites };
        }),
    }),
    {
      name: accountDatabaseName("unsloth_browser_download_sites"),
      version: 1,
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
