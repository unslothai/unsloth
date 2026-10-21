// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

const KEY = "unsloth_model_reasoning_effort";
const MODEL_KEY_PREFIX = `${KEY}::`;
type EffortStorage = Pick<Storage, "getItem" | "setItem" | "key" | "length">;

function browserStorage(): EffortStorage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

function readEfforts(storage: EffortStorage | null): Record<string, string> {
  const efforts: Record<string, string> = {};
  if (!storage) return efforts;
  try {
    const legacy: unknown = JSON.parse(storage.getItem(KEY) ?? "{}");
    if (legacy && typeof legacy === "object" && !Array.isArray(legacy)) {
      Object.assign(
        efforts,
        Object.fromEntries(
          Object.entries(legacy).filter(
            ([, value]) => typeof value === "string" && value.length > 0,
          ),
        ),
      );
    }
  } catch {
    // A broken legacy map must not hide newer entries.
  }
  try {
    for (let index = 0; index < storage.length; index += 1) {
      const key = storage.key(index);
      if (!key?.startsWith(MODEL_KEY_PREFIX)) continue;
      const modelId = key.slice(MODEL_KEY_PREFIX.length);
      const effort = storage.getItem(key);
      if (!modelId || effort === null) continue;
      if (effort) Object.assign(efforts, { [modelId]: effort });
      else delete efforts[modelId];
    }
  } catch {
    // Keep entries already read when storage becomes unavailable.
  }
  return efforts;
}

interface ModelReasoningEffortState {
  effortByModel: Record<string, string>;
  setModelReasoningEffort: (modelId: string, effort: string | null) => void;
  syncFromStorage: () => void;
}

export function createModelReasoningEffortStore(storage: EffortStorage | null) {
  return create<ModelReasoningEffortState>((set) => ({
    effortByModel: readEfforts(storage),
    setModelReasoningEffort: (modelId, effort) =>
      set((state) => {
        if (!modelId) return state;
        const next = { ...state.effortByModel };
        if (effort) Object.assign(next, { [modelId]: effort });
        else delete next[modelId];
        try {
          // Independent writes cannot erase another model's pin.
          // Empty entries prevent a cleared legacy pin from returning.
          storage?.setItem(`${MODEL_KEY_PREFIX}${modelId}`, effort ?? "");
        } catch {
          // Keep the choice for this session.
        }
        return { effortByModel: next };
      }),
    syncFromStorage: () => set({ effortByModel: readEfforts(storage) }),
  }));
}

const storage = browserStorage();
export const useModelReasoningEffortStore = createModelReasoningEffortStore(storage);

/** Return only a pin offered by the current model. */
export function pinnedReasoningEffort(
  modelId: string | null | undefined,
  allowed: readonly string[] | null | undefined,
): string | null {
  if (!modelId) return null;
  const effort = useModelReasoningEffortStore.getState().effortByModel[modelId];
  if (!effort || (allowed && !allowed.includes(effort))) return null;
  return effort;
}

if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.storageArea && event.storageArea !== storage) return;
    if (event.key === KEY || event.key === null || event.key.startsWith(MODEL_KEY_PREFIX)) {
      useModelReasoningEffortStore.getState().syncFromStorage();
    }
  });
}
