// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

import {
  pinnedReasoningEnabled,
  REASONING_ENABLED_KEY as KEY,
  REASONING_ENABLED_PREFIX as MODEL_KEY_PREFIX,
  resolveReasoningEnabledOnLoad,
} from "./reasoning-enabled-pin";

export { pinnedReasoningEnabled, resolveReasoningEnabledOnLoad };
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
