// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ExampleOs = "unix" | "windows";
export type FineTuneAction = "train" | "recipes" | "export";

export const SETTINGS_PANEL_PREFS_STORAGE_KEY = "unsloth_settings_panel_prefs";

export interface SettingsPanelPrefsState {
  // null means "follow whichever model the server has resident".
  agentsAgent: string | null;
  agentsModel: string | null;
  // null means "follow the shell inferred from the client / Studio host".
  agentsOs: ExampleOs | null;
  // The quant carries its model, so remembering a quant never pins a model.
  agentsVariant: string | null;
  agentsVariantModel: string | null;
  setAgentsAgent: (agent: string | null) => void;
  setAgentsModel: (model: string | null, variant: string | null) => void;
  setAgentsOs: (os: ExampleOs) => void;
  setAgentsVariant: (model: string, variant: string) => void;

  // null os means "follow the detected device type".
  apiExampleLang: string | null;
  apiExampleOs: ExampleOs | null;
  apiExampleAgent: string | null;
  setApiExampleLang: (lang: string) => void;
  setApiExampleOs: (os: ExampleOs) => void;
  setApiExampleAgent: (agent: string | null) => void;

  // null mcpOs follows the detected device type.
  mcpAgent: string | null;
  mcpOs: ExampleOs | null;
  setMcpAgent: (agent: string) => void;
  setMcpOs: (os: ExampleOs) => void;

  resourcesLiveUpdates: boolean;
  setResourcesLiveUpdates: (enabled: boolean) => void;

  fineTuneAction: FineTuneAction;
  setFineTuneAction: (action: FineTuneAction) => void;
}

const EXAMPLE_OS_VALUES: ExampleOs[] = ["unix", "windows"];
const FINE_TUNE_VALUES: FineTuneAction[] = ["train", "recipes", "export"];

// Whitelist: localStorage is untyped and a bad record would crash agents-tab path checks.
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
    agentsOs:
      typeof raw.agentsOs === "string" &&
      (EXAMPLE_OS_VALUES as string[]).includes(raw.agentsOs)
        ? (raw.agentsOs as ExampleOs)
        : null,
    // a quant without its model scope cannot be applied.
    agentsVariant: agentsVariantModel ? agentsVariant : null,
    agentsVariantModel: agentsVariant ? agentsVariantModel : null,
    apiExampleLang: text(raw.apiExampleLang),
    apiExampleOs:
      typeof raw.apiExampleOs === "string" &&
      (EXAMPLE_OS_VALUES as string[]).includes(raw.apiExampleOs)
        ? (raw.apiExampleOs as ExampleOs)
        : null,
    apiExampleAgent: text(raw.apiExampleAgent),
    mcpAgent: text(raw.mcpAgent),
    mcpOs: EXAMPLE_OS_VALUES.find((os) => os === raw.mcpOs) ?? null,
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
      agentsOs: null,
      agentsVariant: null,
      agentsVariantModel: null,
      setAgentsAgent: (agentsAgent) => set({ agentsAgent }),
      setAgentsModel: (agentsModel, agentsVariant) =>
        set({ agentsModel, agentsVariant, agentsVariantModel: agentsModel }),
      setAgentsOs: (agentsOs) => set({ agentsOs }),
      setAgentsVariant: (agentsVariantModel, agentsVariant) =>
        set({ agentsVariant, agentsVariantModel }),

      apiExampleLang: null,
      apiExampleOs: null,
      apiExampleAgent: null,
      setApiExampleLang: (apiExampleLang) => set({ apiExampleLang }),
      setApiExampleOs: (apiExampleOs) => set({ apiExampleOs }),
      setApiExampleAgent: (apiExampleAgent) => set({ apiExampleAgent }),

      mcpAgent: null,
      mcpOs: null,
      setMcpAgent: (mcpAgent) => set({ mcpAgent }),
      setMcpOs: (mcpOs) => set({ mcpOs }),

      resourcesLiveUpdates: true,
      setResourcesLiveUpdates: (resourcesLiveUpdates) =>
        set({ resourcesLiveUpdates }),

      fineTuneAction: "train",
      setFineTuneAction: (fineTuneAction) => set({ fineTuneAction }),
    }),
    {
      name: SETTINGS_PANEL_PREFS_STORAGE_KEY,
      // pinned so later shape changes can migrate from this version.
      version: 1,
      // sanitize older records; drop newer versions that may reuse these field names.
      migrate: (persisted, version) =>
        (version < 1 ? persisted : {}) as SettingsPanelPrefsState,
      merge: (persisted, current) => sanitize(persisted, current),
    },
  ),
);
