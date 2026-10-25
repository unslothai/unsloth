// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Named run settings a user keeps for one model, picked from in the run settings panel. Per model,
// not per quant, so "Long context" is offered for every quant. Only "Remember for this model" decides
// what a load applies; these are a list to fill the panel from.

import { useSyncExternalStore } from "react";
import { normalizeModelIdentity } from "./model-identity";
import {
  type PerModelConfig,
  fromStoredPerModelConfig,
  isChatTemplateWithinLimit,
  toStoredPerModelConfig,
} from "./per-model-config";

export const SAVED_RUN_SETTINGS_STORAGE_KEY =
  "unsloth_model_saved_run_settings";
export const SAVED_RUN_SETTINGS_UPDATED_EVENT =
  "unsloth-saved-run-settings-updated";
export const SAVED_RUN_SETTINGS_NAME_MAX = 80;
export const SAVED_RUN_SETTINGS_PER_MODEL_MAX = 24;

export interface SavedRunSettings {
  name: string;
  savedAt: number;
  config: PerModelConfig;
}

type StoredEntry = { name: string; savedAt: number; config: unknown };
type StoredMap = Record<string, StoredEntry[]>;

function storage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

function modelKey(modelId: string): string {
  return normalizeModelIdentity(modelId.trim());
}

export function normalizeSavedRunSettingsName(name: string): string {
  return name.trim().replace(/\s+/g, " ").slice(0, SAVED_RUN_SETTINGS_NAME_MAX);
}

function sameName(a: string, b: string): boolean {
  return a.localeCompare(b, undefined, { sensitivity: "accent" }) === 0;
}

function readMap(): StoredMap {
  const raw = storage()?.getItem(SAVED_RUN_SETTINGS_STORAGE_KEY);
  if (!raw) {
    return {};
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as StoredMap)
      : {};
  } catch {
    return {};
  }
}

function writeMap(map: StoredMap): boolean {
  const store = storage();
  if (!store) {
    return false;
  }
  try {
    store.setItem(SAVED_RUN_SETTINGS_STORAGE_KEY, JSON.stringify(map));
  } catch {
    return false;
  }
  cache.clear();
  // Best-effort, as the per-model store: the write landed, so a host without events still reports it saved.
  if (typeof window?.dispatchEvent === "function") {
    window.dispatchEvent(new Event(SAVED_RUN_SETTINGS_UPDATED_EVENT));
  }
  return true;
}

function decode(entries: unknown): SavedRunSettings[] {
  if (!Array.isArray(entries)) {
    return [];
  }
  const out: SavedRunSettings[] = [];
  for (const entry of entries as StoredEntry[]) {
    const name =
      entry && typeof entry.name === "string"
        ? normalizeSavedRunSettingsName(entry.name)
        : "";
    // A future-schema record stays stored but is not offered: this client cannot apply it.
    const config = name ? fromStoredPerModelConfig(entry.config) : null;
    if (config) {
      out.push({ name, savedAt: Number(entry.savedAt) || 0, config });
    }
  }
  return out.sort((a, b) => a.name.localeCompare(b.name));
}

// Stable per key between writes, as useSyncExternalStore needs.
const cache = new Map<string, SavedRunSettings[]>();

export function listSavedRunSettings(modelId: string): SavedRunSettings[] {
  const key = modelKey(modelId);
  let list = cache.get(key);
  if (!list) {
    list = decode(readMap()[key]);
    cache.set(key, list);
  }
  return list;
}

export function saveRunSettings(
  modelId: string,
  name: string,
  config: PerModelConfig,
): boolean {
  const cleanName = normalizeSavedRunSettingsName(name);
  if (
    !cleanName ||
    (typeof config.chatTemplateOverride === "string" &&
      !isChatTemplateWithinLimit(config.chatTemplateOverride))
  ) {
    return false;
  }
  const map = readMap();
  const key = modelKey(modelId);
  const kept = (Array.isArray(map[key]) ? map[key] : []).filter(
    (entry) =>
      !(
        entry &&
        typeof entry.name === "string" &&
        sameName(entry.name, cleanName)
      ),
  );
  if (kept.length >= SAVED_RUN_SETTINGS_PER_MODEL_MAX) {
    return false;
  }
  map[key] = [
    ...kept,
    {
      name: cleanName,
      savedAt: Date.now(),
      config: toStoredPerModelConfig(config),
    },
  ];
  return writeMap(map);
}

export function deleteSavedRunSettings(modelId: string, name: string): boolean {
  const map = readMap();
  const key = modelKey(modelId);
  const entries = Array.isArray(map[key]) ? map[key] : [];
  const kept = entries.filter(
    (entry) =>
      !(entry && typeof entry.name === "string" && sameName(entry.name, name)),
  );
  if (kept.length === entries.length) {
    return true;
  }
  if (kept.length) {
    map[key] = kept;
  } else {
    delete map[key];
  }
  return writeMap(map);
}

export function findSavedRunSettings(
  list: readonly SavedRunSettings[],
  name: string,
): SavedRunSettings | null {
  const cleanName = normalizeSavedRunSettingsName(name);
  return cleanName
    ? (list.find((entry) => sameName(entry.name, cleanName)) ?? null)
    : null;
}

function subscribe(onChange: () => void): () => void {
  const onStorage = (event: StorageEvent) => {
    if (event.key === null || event.key === SAVED_RUN_SETTINGS_STORAGE_KEY) {
      cache.clear();
      onChange();
    }
  };
  window.addEventListener(SAVED_RUN_SETTINGS_UPDATED_EVENT, onChange);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(SAVED_RUN_SETTINGS_UPDATED_EVENT, onChange);
    window.removeEventListener("storage", onStorage);
  };
}

const EMPTY: SavedRunSettings[] = [];

export function useSavedRunSettings(modelId: string): SavedRunSettings[] {
  return useSyncExternalStore(
    subscribe,
    () => listSavedRunSettings(modelId),
    () => EMPTY,
  );
}
