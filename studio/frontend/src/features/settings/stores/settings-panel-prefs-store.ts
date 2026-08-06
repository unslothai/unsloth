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
  // the quant carries the model it was picked for, so remembering a quant alone
  // never pins a model the tab would otherwise keep following.
  agentsVariant: string | null;
  agentsVariantModel: string | null;
  setAgentsAgent: (agent: string | null) => void;
  setAgentsModel: (model: string | null, variant: string | null) => void;
  setAgentsVariant: (model: string, variant: string) => void;
  // Drops the quant without touching the model, for the case where the repo
  // stops offering a remembered quant but the model itself is still valid.
  clearAgentsVariant: () => void;

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

const EXAMPLE_OS_VALUES: ExampleOs[] = ["unix", "windows"];
const FINE_TUNE_VALUES: FineTuneAction[] = ["train", "recipes", "export"];

// localStorage is untyped at runtime and these reach `.toLowerCase()` and the
// path checks in agents-tab, so a bad record would take the app down. The
// default merge also spreads the blob over the actions themselves. Whitelist.
function text(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function oneOf<T extends string>(value: unknown, allowed: T[], fallback: T): T {
  return typeof value === "string" && (allowed as string[]).includes(value)
    ? (value as T)
    : fallback;
}

function sanitize(
  persisted: unknown,
  current: SettingsPanelPrefsState,
): SettingsPanelPrefsState {
  const raw = (
    persisted && typeof persisted === "object" ? persisted : {}
  ) as Record<string, unknown>;
  const agentsModel = text(raw.agentsModel);
  const agentsVariantModel = text(raw.agentsVariantModel);
  const agentsVariant = text(raw.agentsVariant);
  return {
    ...current,
    agentsAgent: text(raw.agentsAgent),
    agentsModel,
    // A quant with no model to scope it to can never be applied.
    agentsVariant: agentsVariantModel ? agentsVariant : null,
    agentsVariantModel: agentsVariant ? agentsVariantModel : null,
    apiExampleLang: text(raw.apiExampleLang),
    apiExampleOs:
      typeof raw.apiExampleOs === "string" &&
      (EXAMPLE_OS_VALUES as string[]).includes(raw.apiExampleOs)
        ? (raw.apiExampleOs as ExampleOs)
        : null,
    apiExampleAgent: text(raw.apiExampleAgent),
    resourcesLiveUpdates:
      typeof raw.resourcesLiveUpdates === "boolean"
        ? raw.resourcesLiveUpdates
        : true,
    fineTuneAction: oneOf(raw.fineTuneAction, FINE_TUNE_VALUES, "train"),
  };
}

export const useSettingsPanelPrefsStore = create<SettingsPanelPrefsState>()(
  persist(
    (set) => ({
      agentsAgent: null,
      agentsModel: null,
      agentsVariant: null,
      agentsVariantModel: null,
      setAgentsAgent: (agentsAgent) => set({ agentsAgent }),
      // picking a model carries its quant, and clearing it clears that quant.
      setAgentsModel: (agentsModel, agentsVariant) =>
        set({ agentsModel, agentsVariant, agentsVariantModel: agentsModel }),
      setAgentsVariant: (agentsVariantModel, agentsVariant) =>
        set({ agentsVariant, agentsVariantModel }),
      clearAgentsVariant: () =>
        set({ agentsVariant: null, agentsVariantModel: null }),

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
    {
      name: SETTINGS_PANEL_PREFS_STORAGE_KEY,
      // Pinned so a later shape change has somewhere to migrate from.
      version: 1,
      // Older records share these field names, so the sanitiser is enough. A
      // newer one is dropped: it may reuse the names with different meaning.
      migrate: (persisted, version) =>
        (version < 1 ? persisted : {}) as SettingsPanelPrefsState,
      merge: (persisted, current) => sanitize(persisted, current),
    },
  ),
);
