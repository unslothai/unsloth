// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ExampleOs = "unix" | "windows";
export type FineTuneAction = "train" | "recipes" | "export";

export const SETTINGS_PANEL_PREFS_STORAGE_KEY = "unsloth_settings_panel_prefs";

// settings dialog picks that renderTab() used to drop on every tab switch.
export interface SettingsPanelPrefsState {
  // null model means "follow whichever model the server has resident".
  agentsAgent: string | null;
  agentsModel: string | null;
  agentsVariant: string | null;
  setAgentsAgent: (agent: string | null) => void;
  setAgentsModel: (model: string | null, variant: string | null) => void;
  setAgentsVariant: (variant: string | null) => void;

  // null os means "follow the detected device type".
  apiExampleLang: string | null;
  apiExampleOs: ExampleOs | null;
  apiExampleAgent: string | null;
  setApiExampleLang: (lang: string) => void;
  setApiExampleOs: (os: ExampleOs) => void;
  setApiExampleAgent: (agent: string | null) => void;

  resourcesLiveUpdates: boolean;
  setResourcesLiveUpdates: (enabled: boolean) => void;

  fineTuneAction: FineTuneAction;
  setFineTuneAction: (action: FineTuneAction) => void;
}

export const useSettingsPanelPrefsStore = create<SettingsPanelPrefsState>()(
  persist(
    (set) => ({
      agentsAgent: null,
      agentsModel: null,
      agentsVariant: null,
      setAgentsAgent: (agentsAgent) => set({ agentsAgent }),
      // a quant belongs to its repo, so the two only ever move together.
      setAgentsModel: (agentsModel, agentsVariant) =>
        set({ agentsModel, agentsVariant }),
      setAgentsVariant: (agentsVariant) => set({ agentsVariant }),

      apiExampleLang: null,
      apiExampleOs: null,
      apiExampleAgent: null,
      setApiExampleLang: (apiExampleLang) => set({ apiExampleLang }),
      setApiExampleOs: (apiExampleOs) => set({ apiExampleOs }),
      setApiExampleAgent: (apiExampleAgent) => set({ apiExampleAgent }),

      resourcesLiveUpdates: true,
      setResourcesLiveUpdates: (resourcesLiveUpdates) =>
        set({ resourcesLiveUpdates }),

      fineTuneAction: "train",
      setFineTuneAction: (fineTuneAction) => set({ fineTuneAction }),
    }),
    { name: SETTINGS_PANEL_PREFS_STORAGE_KEY },
  ),
);
