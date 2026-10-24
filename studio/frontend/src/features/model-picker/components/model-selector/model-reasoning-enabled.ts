// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

const KEY = "unsloth_model_reasoning_enabled";
const MODEL_KEY_PREFIX = `${KEY}::`;
type EnabledStorage = Pick<Storage, "getItem" | "setItem" | "key" | "length">;

function browserStorage(): EnabledStorage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

function parseEnabled(raw: string | null): boolean | undefined {
  if (raw === "true") return true;
  if (raw === "false") return false;
  return undefined;
}

function readEnabled(storage: EnabledStorage | null): Record<string, boolean> {
  const enabled: Record<string, boolean> = {};
  if (!storage) return enabled;
  try {
    const legacy: unknown = JSON.parse(storage.getItem(KEY) ?? "{}");
    if (legacy && typeof legacy === "object" && !Array.isArray(legacy)) {
      for (const [modelId, value] of Object.entries(legacy)) {
        const parsed =
          typeof value === "boolean" ? value : parseEnabled(String(value));
        if (parsed !== undefined) Object.assign(enabled, { [modelId]: parsed });
      }
    }
  } catch {
    // A broken legacy map must not hide newer entries.
  }
  try {
    for (let index = 0; index < storage.length; index += 1) {
      const key = storage.key(index);
      if (!key?.startsWith(MODEL_KEY_PREFIX)) continue;
      const modelId = key.slice(MODEL_KEY_PREFIX.length);
      if (!modelId) continue;
      const parsed = parseEnabled(storage.getItem(key));
      if (parsed === undefined) delete enabled[modelId];
      else Object.assign(enabled, { [modelId]: parsed });
    }
  } catch {
    // Keep entries already read when storage becomes unavailable.
  }
  return enabled;
}

interface ModelReasoningEnabledState {
  enabledByModel: Record<string, boolean>;
  setModelReasoningEnabled: (modelId: string, enabled: boolean | null) => void;
  syncFromStorage: () => void;
}

export function createModelReasoningEnabledStore(
  storage: EnabledStorage | null,
) {
  return create<ModelReasoningEnabledState>((set) => ({
    enabledByModel: readEnabled(storage),
    setModelReasoningEnabled: (modelId, enabled) =>
      set((state) => {
        if (!modelId) return state;
        const next = { ...state.enabledByModel };
        if (enabled === null) delete next[modelId];
        else Object.assign(next, { [modelId]: enabled });
        try {
          storage?.setItem(
            `${MODEL_KEY_PREFIX}${modelId}`,
            enabled === null ? "" : enabled ? "true" : "false",
          );
        } catch {
          // Keep the choice for this session.
        }
        return { enabledByModel: next };
      }),
    syncFromStorage: () => set({ enabledByModel: readEnabled(storage) }),
  }));
}

const storage = browserStorage();
export const useModelReasoningEnabledStore =
  createModelReasoningEnabledStore(storage);

/** Last thinking on/off the user chose for this checkpoint, or null if they never chose. */
export function pinnedReasoningEnabled(
  modelId: string | null | undefined,
): boolean | null {
  if (!modelId) return null;
  const value =
    useModelReasoningEnabledStore.getState().enabledByModel[modelId];
  return value === undefined ? null : value;
}

/**
 * Thinking on/off after a load. A per-model pin outranks the family default so ejecting
 * and reloading the same checkpoint keeps Off. Always-on / effort-only models cannot
 * honor Off. persist:false toggles never write a pin.
 */
export function resolveReasoningEnabledOnLoad(options: {
  reasoningAlwaysOn: boolean;
  supportsReasoning: boolean;
  supportsReasoningOff: boolean;
  pinned: boolean | null;
  reloadingSameModel: boolean;
  previousEnabled: boolean;
  reasoningDefault: boolean;
}): boolean {
  if (options.reasoningAlwaysOn) return true;
  if (!options.supportsReasoning) return options.reasoningDefault;
  if (!options.supportsReasoningOff) return true;
  if (options.pinned !== null) return options.pinned;
  if (options.reloadingSameModel) return options.previousEnabled;
  return options.reasoningDefault;
}

if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.storageArea && event.storageArea !== storage) return;
    if (
      event.key === KEY ||
      event.key === null ||
      event.key.startsWith(MODEL_KEY_PREFIX)
    ) {
      useModelReasoningEnabledStore.getState().syncFromStorage();
    }
  });
}
