// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Named run settings per model, across quants, and which one each model + quant loads with. The selected set is
// mirrored into the per-model record, which is what a load (and the API) applies.

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
export const SELECTED_RUN_SETTINGS_STORAGE_KEY =
  "unsloth_model_selected_run_settings";
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
/** model + quant key -> set name, or null for Default. Absent = never chosen. */
type SelectionMap = Record<string, string | null>;

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

function selectionKey(modelId: string, ggufVariant: string | null | undefined) {
  return JSON.stringify([
    modelKey(modelId),
    (ggufVariant ?? "").trim().toLowerCase(),
  ]);
}

function selectionPrefix(modelId: string): string {
  return `${JSON.stringify([modelKey(modelId)]).slice(0, -1)},`;
}

export function normalizeSavedRunSettingsName(name: string): string {
  return name.trim().replace(/\s+/g, " ").slice(0, SAVED_RUN_SETTINGS_NAME_MAX);
}

function sameName(a: string, b: string): boolean {
  return a.localeCompare(b, undefined, { sensitivity: "accent" }) === 0;
}

function readJson<T extends object>(key: string): T {
  const raw = storage()?.getItem(key);
  if (!raw) {
    return {} as T;
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as T)
      : ({} as T);
  } catch {
    return {} as T;
  }
}

function writeJson(key: string, value: object): boolean {
  const store = storage();
  if (!store) {
    return false;
  }
  try {
    store.setItem(key, JSON.stringify(value));
  } catch {
    return false;
  }
  cache.clear();
  // Best-effort: the write already landed.
  if (typeof window?.dispatchEvent === "function") {
    window.dispatchEvent(new Event(SAVED_RUN_SETTINGS_UPDATED_EVENT));
  }
  return true;
}

const readMap = () => readJson<StoredMap>(SAVED_RUN_SETTINGS_STORAGE_KEY);
const writeMap = (map: StoredMap) =>
  writeJson(SAVED_RUN_SETTINGS_STORAGE_KEY, map);
const readSelections = () =>
  readJson<SelectionMap>(SELECTED_RUN_SETTINGS_STORAGE_KEY);
const writeSelections = (map: SelectionMap) =>
  writeJson(SELECTED_RUN_SETTINGS_STORAGE_KEY, map);
// The quant as the picker spells it, by selection key: a server override key keeps that case, and
// an exact key wins over a folded one, so a lower-cased write would lose to a stale entry.
const SELECTED_VARIANTS_STORAGE_KEY =
  "unsloth_model_selected_run_settings_variants";
const readSelectedVariants = () =>
  readJson<Record<string, string>>(SELECTED_VARIANTS_STORAGE_KEY);

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
    // Future-schema records stay stored but are not offered.
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

function entriesFor(map: StoredMap, key: string): StoredEntry[] {
  return Array.isArray(map[key]) ? map[key] : [];
}

function named(entry: StoredEntry | null | undefined, name: string): boolean {
  return Boolean(
    entry && typeof entry.name === "string" && sameName(entry.name, name),
  );
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
  const kept = entriesFor(map, key).filter((entry) => !named(entry, cleanName));
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

function selectionsNaming(
  selections: SelectionMap,
  modelId: string,
  name: string,
): string[] {
  const prefix = selectionPrefix(modelId);
  return Object.keys(selections).filter((key) => {
    const value = selections[key];
    return (
      key.startsWith(prefix) &&
      typeof value === "string" &&
      sameName(value, name)
    );
  });
}

/** False when ``to`` is blank or another set already has that name. */
export function renameSavedRunSettings(
  modelId: string,
  from: string,
  to: string,
): boolean {
  const cleanName = normalizeSavedRunSettingsName(to);
  const map = readMap();
  const key = modelKey(modelId);
  const entries = entriesFor(map, key);
  const entry = entries.find((candidate) => named(candidate, from));
  if (
    !(cleanName && entry) ||
    entries.some((other) => other !== entry && named(other, cleanName))
  ) {
    return false;
  }
  entry.name = cleanName;
  if (!writeMap(map)) {
    return false;
  }
  const selections = readSelections();
  const renamed = selectionsNaming(selections, modelId, from);
  for (const selected of renamed) {
    selections[selected] = cleanName;
  }
  return renamed.length === 0 || writeSelections(selections);
}

/** Drops the set; quants that loaded with it go to Default. */
export function deleteSavedRunSettings(modelId: string, name: string): boolean {
  const map = readMap();
  const key = modelKey(modelId);
  const entries = entriesFor(map, key);
  const kept = entries.filter((entry) => !named(entry, name));
  if (kept.length !== entries.length) {
    if (kept.length > 0) {
      map[key] = kept;
    } else {
      delete map[key];
    }
    if (!writeMap(map)) {
      return false;
    }
  }
  const selections = readSelections();
  const dropped = selectionsNaming(selections, modelId, name);
  // Default, not unchosen: an unchosen quant would adopt its remembered record back into a set.
  for (const selected of dropped) {
    selections[selected] = null;
  }
  return dropped.length === 0 || writeSelections(selections);
}

/** The quants of this model that load with the set ("" is a model without quants). */
export function quantsSelectingRunSettings(
  modelId: string,
  name: string,
): string[] {
  const variants = readSelectedVariants();
  return selectionsNaming(readSelections(), modelId, name).map((key) => {
    if (typeof variants[key] === "string") {
      return variants[key];
    }
    const parsed: unknown = JSON.parse(key);
    return Array.isArray(parsed) && typeof parsed[1] === "string"
      ? parsed[1]
      : "";
  });
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

/** The set this model + quant loads with, null for Default, undefined when never chosen. */
export function getSelectedRunSettings(
  modelId: string,
  ggufVariant: string | null | undefined,
): string | null | undefined {
  const value = readSelections()[selectionKey(modelId, ggufVariant)];
  return typeof value === "string" || value === null ? value : undefined;
}

export function setSelectedRunSettings(
  modelId: string,
  ggufVariant: string | null | undefined,
  name: string | null,
): boolean {
  const key = selectionKey(modelId, ggufVariant);
  const variants = readSelectedVariants();
  const spelled = (ggufVariant ?? "").trim();
  if (variants[key] !== spelled) {
    variants[key] = spelled;
    writeJson(SELECTED_VARIANTS_STORAGE_KEY, variants);
  }
  const selections = readSelections();
  if (selections[key] === name) {
    return true;
  }
  selections[key] = name;
  return writeSelections(selections);
}

/** "<model> custom settings", numbered past the names already taken. */
export function newRunSettingsName(
  list: readonly SavedRunSettings[],
  modelName: string,
): string {
  const base = `${modelName} custom settings`;
  let name = withSuffix(base, "");
  for (let n = 2; findSavedRunSettings(list, name); n++) {
    name = withSuffix(base, ` ${n}`);
  }
  return name;
}

// Trims the base, not the suffix: a cut suffix would name the same set forever.
function withSuffix(base: string, suffix: string): string {
  return normalizeSavedRunSettingsName(
    `${normalizeSavedRunSettingsName(base).slice(0, SAVED_RUN_SETTINGS_NAME_MAX - suffix.length)}${suffix}`,
  );
}

// Names earlier builds gave sets: an adopted record, then "Custom setting for <model>[ N]".
const LEGACY_ADOPTED_NAME = "Saved settings";

function legacyRename(
  name: string,
  list: readonly SavedRunSettings[],
  modelName: string,
): string | null {
  if (sameName(name, LEGACY_ADOPTED_NAME)) {
    return newRunSettingsName(list, modelName);
  }
  const old = `Custom setting for ${modelName}`;
  const suffix = name.slice(old.length);
  if (!sameName(name.slice(0, old.length), old) || !/^( \d+)?$/.test(suffix)) {
    return null;
  }
  return withSuffix(`${modelName} custom settings`, suffix);
}

export function hasLegacyRunSettings(
  list: readonly SavedRunSettings[],
  modelName: string,
): boolean {
  return list.some(
    (entry) => legacyRename(entry.name, list, modelName) !== null,
  );
}

export function renameLegacyRunSettings(
  modelId: string,
  modelName: string,
): string[] {
  const renamed: string[] = [];
  for (const { name } of listSavedRunSettings(modelId)) {
    const to = legacyRename(name, listSavedRunSettings(modelId), modelName);
    if (to !== null && renameSavedRunSettings(modelId, name, to)) {
      renamed.push(to);
    }
  }
  return renamed;
}

/** Remembered settings saved before named sets existed: list them as a set and select it. Returns its name. */
export function adoptRememberedRunSettings(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig,
  modelName: string,
): string | null {
  const stored = JSON.stringify(toStoredPerModelConfig(config));
  const list = listSavedRunSettings(modelId);
  let name = list.find(
    (entry) => JSON.stringify(toStoredPerModelConfig(entry.config)) === stored,
  )?.name;
  if (!name) {
    name = newRunSettingsName(list, modelName);
    if (!saveRunSettings(modelId, name, config)) {
      return null;
    }
  }
  return setSelectedRunSettings(modelId, ggufVariant, name) ? name : null;
}

export function subscribeSavedRunSettings(onChange: () => void): () => void {
  // No listener ran while unsubscribed: another tab may have written.
  cache.clear();
  const onStorage = (event: StorageEvent) => {
    if (
      event.key === null ||
      event.key === SAVED_RUN_SETTINGS_STORAGE_KEY ||
      event.key === SELECTED_RUN_SETTINGS_STORAGE_KEY
    ) {
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
    subscribeSavedRunSettings,
    () => listSavedRunSettings(modelId),
    () => EMPTY,
  );
}

export function useSelectedRunSettings(
  modelId: string,
  ggufVariant: string | null | undefined,
): string | null | undefined {
  return useSyncExternalStore(
    subscribeSavedRunSettings,
    () => getSelectedRunSettings(modelId, ggufVariant),
    () => undefined,
  );
}
