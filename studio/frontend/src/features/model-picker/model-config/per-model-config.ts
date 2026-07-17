// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ggufVariantFromStorageKey,
  modelIdFromStorageKey,
  modelStorageKey,
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
} from "./model-identity";

export interface PerModelConfig {
  customContextLength: number | null;
  maxSeqLength: number | null;
  kvCacheDtype: string | null;
  speculativeType: string | null;
  specDraftNMax: number | null;
  tensorParallel: boolean;
  chatTemplateOverride: string | null;
}

export const DEFAULT_PER_MODEL_CONFIG: PerModelConfig = {
  customContextLength: null,
  maxSeqLength: null,
  kvCacheDtype: null,
  speculativeType: null,
  specDraftNMax: null,
  tensorParallel: false,
  chatTemplateOverride: null,
};

export const MAX_SEQ_LENGTH_MIN = 128;
export const MAX_SEQ_LENGTH_MAX = 1048576;
export const MAX_SEQ_LENGTH_STEP = 128;
export const CONTEXT_LENGTH_MIN = 128;

export const KV_CACHE_DTYPES = ["bf16", "q8_0", "q5_1", "q4_1"] as const;
const VALID_KV_CACHE_DTYPES = new Set<string>(KV_CACHE_DTYPES);

export const SPECULATIVE_TYPES = [
  "auto",
  "mtp",
  "ngram",
  "mtp+ngram",
  "off",
] as const;
export const MTP_SPECULATIVE_TYPES: ReadonlySet<string> = new Set([
  "mtp",
  "mtp+ngram",
]);

const STORAGE_KEY = "unsloth_model_configs";
const LEGACY_STORAGE_KEY = "unsloth_load_settings";
const LEGACY_MIGRATION_FLAG = "unsloth_model_configs_migrated";
const STORAGE_SCHEMA_VERSION = 1;
const MAX_ENTRIES = 500;
const MAX_PER_MODEL_CONFIG_STORAGE_BYTES = 1024 * 1024;
export const MAX_CHAT_TEMPLATE_BYTES = 65_536;

type StoredPerModelConfig = PerModelConfig & {
  version: typeof STORAGE_SCHEMA_VERSION;
};
type StoredMap = Record<string, PerModelConfig | StoredPerModelConfig>;
type RawConfig = Partial<PerModelConfig> & { version?: unknown };

const STORED_CONFIG_FIELDS = new Set([
  "version",
  "customContextLength",
  "maxSeqLength",
  "kvCacheDtype",
  "speculativeType",
  "specDraftNMax",
  "tensorParallel",
  "chatTemplateOverride",
]);

function canonicalizeSpeculativeType(value: string): string | null {
  const s = value.trim().toLowerCase();
  if (!s) {
    return null;
  }
  if (s === "auto" || s === "default") {
    return "auto";
  }
  if (s === "off") {
    return "off";
  }
  if (s === "mtp" || s === "draft-mtp") {
    return "mtp";
  }
  if (s === "ngram" || s === "ngram-mod" || s === "ngram-simple") {
    return "ngram";
  }
  if (s === "mtp+ngram") {
    return "mtp+ngram";
  }
  return null;
}

export function normalizeMaxSeqLength(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  const snapped = Math.round(value / MAX_SEQ_LENGTH_STEP) * MAX_SEQ_LENGTH_STEP;
  return Math.max(MAX_SEQ_LENGTH_MIN, Math.min(MAX_SEQ_LENGTH_MAX, snapped));
}

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function serializedByteLength(value: string): number {
  return typeof TextEncoder !== "undefined"
    ? new TextEncoder().encode(value).byteLength
    : value.length;
}

export function chatTemplateByteLength(value: string): number {
  return serializedByteLength(value);
}

export function isChatTemplateWithinLimit(value: string): boolean {
  return chatTemplateByteLength(value) <= MAX_CHAT_TEMPLATE_BYTES;
}

function serializedMapSize(map: StoredMap): number {
  return serializedByteLength(JSON.stringify(map));
}

function serializedMapEntrySize(key: string, value: StoredMap[string]): number {
  return (
    serializedByteLength(JSON.stringify(key)) +
    1 +
    serializedByteLength(JSON.stringify(value))
  );
}

function deleteOldestEvictableEntry(
  map: StoredMap,
  protectedKeys?: ReadonlySet<string>,
): { key: string; value: StoredMap[string] } | null {
  for (const key of Object.keys(map)) {
    // Never evict a future-schema entry an older client cannot interpret,
    // matching the save/delete guards.
    if (
      protectedKeys?.has(key) ||
      storedConfigVersion(map[key]) > STORAGE_SCHEMA_VERSION
    ) {
      continue;
    }
    const value = map[key];
    delete map[key];
    return { key, value };
  }
  return null;
}

function enforceStorageBudget(
  map: StoredMap,
  protectedKeys?: ReadonlySet<string>,
): boolean {
  let entryCount = Object.keys(map).length;
  while (entryCount > MAX_ENTRIES) {
    if (!deleteOldestEvictableEntry(map, protectedKeys)) {
      return false;
    }
    entryCount -= 1;
  }
  let bytes = serializedMapSize(map);
  while (bytes > MAX_PER_MODEL_CONFIG_STORAGE_BYTES) {
    const removed = deleteOldestEvictableEntry(map, protectedKeys);
    if (!removed) {
      return false;
    }
    bytes -=
      serializedMapEntrySize(removed.key, removed.value) +
      (entryCount > 1 ? 1 : 0);
    entryCount -= 1;
  }
  return true;
}

function storedConfigVersion(raw: unknown): number {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return 0;
  }
  const version = (raw as RawConfig).version;
  return typeof version === "number" && Number.isFinite(version) ? version : 0;
}

let legacyMigrationChecked = false;

function parseLegacyModelKey(
  key: string,
): { modelId: string; ggufVariant: string | null } | null {
  const separator = key.lastIndexOf("::");
  if (separator >= 0) {
    const modelId = key.slice(0, separator);
    return modelId
      ? { modelId, ggufVariant: key.slice(separator + 2) || null }
      : null;
  }
  return key ? { modelId: key, ggufVariant: null } : null;
}

function legacyEntryToConfig(raw: Record<string, unknown>): PerModelConfig {
  return normalizeV1({
    customContextLength:
      typeof raw.contextLength === "number" ? raw.contextLength : null,
    maxSeqLength: null,
    kvCacheDtype:
      typeof raw.kvCacheDtype === "string" ? raw.kvCacheDtype : null,
    speculativeType:
      typeof raw.speculativeType === "string" ? raw.speculativeType : null,
    specDraftNMax:
      typeof raw.specDraftNMax === "number" ? raw.specDraftNMax : null,
    tensorParallel:
      typeof raw.tensorParallel === "boolean" ? raw.tensorParallel : false,
    chatTemplateOverride: null,
  });
}

function mergeLegacyEntries(
  map: StoredMap,
  legacy: Record<string, unknown>,
): string[] {
  const addedKeys: string[] = [];
  for (const [legacyKey, value] of Object.entries(legacy)) {
    if (!value || typeof value !== "object") {
      continue;
    }
    const parsedKey = parseLegacyModelKey(legacyKey);
    if (!parsedKey) {
      continue;
    }
    const migrated = legacyEntryToConfig(value as Record<string, unknown>);
    const key = modelStorageKey(parsedKey.modelId, parsedKey.ggufVariant);
    if (isDefaultConfig(migrated) || Object.hasOwn(map, key)) {
      continue;
    }
    map[key] = toStoredConfig(migrated);
    addedKeys.push(key);
  }
  return addedKeys;
}

function migrateLegacyLoadSettingsOnce(): void {
  if (legacyMigrationChecked || !canUseStorage()) {
    return;
  }
  legacyMigrationChecked = true;
  try {
    if (localStorage.getItem(LEGACY_MIGRATION_FLAG)) {
      return;
    }
    let legacy: unknown = null;
    try {
      legacy = JSON.parse(localStorage.getItem(LEGACY_STORAGE_KEY) ?? "null");
    } catch {
      legacy = null;
    }
    if (!legacy || typeof legacy !== "object" || Array.isArray(legacy)) {
      localStorage.setItem(LEGACY_MIGRATION_FLAG, "1");
      return;
    }
    const map = readMapRaw();
    const migratedKeys = mergeLegacyEntries(
      map,
      legacy as Record<string, unknown>,
    );
    if (migratedKeys.length === 0) {
      localStorage.setItem(LEGACY_MIGRATION_FLAG, "1");
      return;
    }
    // Protect the just-migrated entries during eviction. If the budget cannot
    // fit them (e.g. storage is full of future-schema records an older client
    // cannot evict), leave the flag unset so migration retries once space frees
    // up rather than marking it complete and dropping the migrated config.
    if (!enforceStorageBudget(map, new Set(migratedKeys))) {
      return;
    }
    if (writeMap(map)) {
      localStorage.setItem(LEGACY_MIGRATION_FLAG, "1");
    }
  } catch (err) {
    console.warn("Failed to migrate legacy load settings:", err);
  }
}

function readMapRaw(): StoredMap {
  if (!canUseStorage()) {
    return {};
  }
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    return parsed as StoredMap;
  } catch {
    return {};
  }
}

function readMap(): StoredMap {
  migrateLegacyLoadSettingsOnce();
  return readMapRaw();
}

function writeMap(map: StoredMap): boolean {
  if (!canUseStorage()) {
    return false;
  }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
    return true;
  } catch (err) {
    console.warn("Failed to persist per-model config:", err);
    return false;
  }
}

function warnDroppedFields(raw: Record<string, unknown>, version: number): void {
  if (!import.meta.env?.DEV) {
    return;
  }
  const dropped = Object.keys(raw).filter(
    (key) => !STORED_CONFIG_FIELDS.has(key),
  );
  if (dropped.length > 0) {
    console.warn("Dropped unknown per-model config fields:", dropped);
  }
  if (version > STORAGE_SCHEMA_VERSION) {
    console.warn("Per-model config schema is newer than this app:", version);
  }
}

function normalizeV1(partial: RawConfig): PerModelConfig {
  const rawSpecType =
    typeof partial.speculativeType === "string"
      ? canonicalizeSpeculativeType(partial.speculativeType)
      : null;
  const speculativeType = rawSpecType ?? DEFAULT_PER_MODEL_CONFIG.speculativeType;
  const specDraftNMax =
    speculativeType != null &&
    MTP_SPECULATIVE_TYPES.has(speculativeType) &&
    typeof partial.specDraftNMax === "number" &&
    Number.isFinite(partial.specDraftNMax)
      ? Math.max(1, Math.min(16, Math.round(partial.specDraftNMax)))
      : null;
  return {
    customContextLength:
      typeof partial.customContextLength === "number" &&
      Number.isFinite(partial.customContextLength) &&
      partial.customContextLength > 0
        ? Math.max(CONTEXT_LENGTH_MIN, Math.floor(partial.customContextLength))
        : null,
    maxSeqLength: normalizeMaxSeqLength(partial.maxSeqLength),
    kvCacheDtype:
      typeof partial.kvCacheDtype === "string" &&
      VALID_KV_CACHE_DTYPES.has(partial.kvCacheDtype)
        ? partial.kvCacheDtype
        : null,
    speculativeType,
    specDraftNMax,
    tensorParallel:
      typeof partial.tensorParallel === "boolean"
        ? partial.tensorParallel
        : DEFAULT_PER_MODEL_CONFIG.tensorParallel,
    chatTemplateOverride:
      typeof partial.chatTemplateOverride === "string" &&
      isChatTemplateWithinLimit(partial.chatTemplateOverride)
        ? partial.chatTemplateOverride
        : null,
  };
}

function normalize(raw: unknown): PerModelConfig {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return normalizeV1({});
  }
  const partial = raw as RawConfig;
  const version =
    typeof partial.version === "number" && Number.isFinite(partial.version)
      ? partial.version
      : 0;
  warnDroppedFields(raw as Record<string, unknown>, version);
  return normalizeV1(partial);
}

function toStoredConfig(config: PerModelConfig): StoredPerModelConfig {
  return {
    version: STORAGE_SCHEMA_VERSION,
    ...normalize(config),
  };
}

function legacyModelStorageKey(
  modelId: string,
  ggufVariant?: string | null,
): string {
  return `${modelId}::${ggufVariant ?? ""}`;
}

function storageKeysForModelVariant(
  modelId: string,
  ggufVariant?: string | null,
): string[] {
  const key = modelStorageKey(modelId, ggufVariant);
  const legacyKey = legacyModelStorageKey(modelId, ggufVariant);
  return key === legacyKey ? [key] : [key, legacyKey];
}

function configKeyMatchesModelVariant(
  key: string,
  modelId: string,
  ggufVariant?: string | null,
): boolean {
  const storedModelId = modelIdFromStorageKey(key);
  if (!storedModelId) {
    return false;
  }
  return (
    normalizeModelIdentity(storedModelId) === normalizeModelIdentity(modelId) &&
    normalizeGgufVariantIdentity(ggufVariantFromStorageKey(key)) ===
      normalizeGgufVariantIdentity(ggufVariant)
  );
}

function findConfigKeyForModelVariant(
  map: StoredMap,
  modelId: string,
  ggufVariant?: string | null,
): string | null {
  for (const key of storageKeysForModelVariant(modelId, ggufVariant)) {
    if (Object.hasOwn(map, key)) {
      return key;
    }
  }
  for (const key of Object.keys(map)) {
    if (configKeyMatchesModelVariant(key, modelId, ggufVariant)) {
      return key;
    }
  }
  return null;
}

function hasFutureConfigForModelVariant(
  map: StoredMap,
  modelId: string,
  ggufVariant?: string | null,
): boolean {
  for (const key of Object.keys(map)) {
    if (
      configKeyMatchesModelVariant(key, modelId, ggufVariant) &&
      storedConfigVersion(map[key]) > STORAGE_SCHEMA_VERSION
    ) {
      return true;
    }
  }
  return false;
}

function deleteConfigEntriesForModelVariant(
  map: StoredMap,
  modelId: string,
  ggufVariant?: string | null,
): boolean {
  let changed = false;
  for (const key of Object.keys(map)) {
    if (!configKeyMatchesModelVariant(key, modelId, ggufVariant)) {
      continue;
    }
    delete map[key];
    changed = true;
  }
  return changed;
}

function loadPerModelConfig(
  modelId: string,
  ggufVariant?: string | null,
): PerModelConfig | null {
  const map = readMap();
  const key = findConfigKeyForModelVariant(map, modelId, ggufVariant);
  if (!key) {
    return null;
  }
  // Never apply a future-schema record an older client cannot interpret,
  // matching the save/delete/evict guards.
  if (storedConfigVersion(map[key]) > STORAGE_SCHEMA_VERSION) {
    return null;
  }
  return normalize(map[key]);
}

export function isDefaultConfig(config: PerModelConfig): boolean {
  return (
    config.customContextLength == null &&
    config.maxSeqLength == null &&
    (config.kvCacheDtype ?? null) === DEFAULT_PER_MODEL_CONFIG.kvCacheDtype &&
    config.speculativeType === DEFAULT_PER_MODEL_CONFIG.speculativeType &&
    config.specDraftNMax == null &&
    Boolean(config.tensorParallel) ===
      Boolean(DEFAULT_PER_MODEL_CONFIG.tensorParallel) &&
    (config.chatTemplateOverride ?? null) === null
  );
}

export function savePerModelConfig(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig,
): boolean {
  if (
    typeof config.chatTemplateOverride === "string" &&
    !isChatTemplateWithinLimit(config.chatTemplateOverride)
  ) {
    return false;
  }
  const normalized = normalize(config);
  const map = readMap();
  if (hasFutureConfigForModelVariant(map, modelId, ggufVariant)) {
    return false;
  }
  if (isDefaultConfig(normalized)) {
    const changed = deleteConfigEntriesForModelVariant(
      map,
      modelId,
      ggufVariant,
    );
    return changed ? writeMap(map) : true;
  }
  const [key] = storageKeysForModelVariant(modelId, ggufVariant);
  deleteConfigEntriesForModelVariant(map, modelId, ggufVariant);
  map[key] = toStoredConfig(normalized);
  if (!enforceStorageBudget(map, new Set([key]))) {
    return false;
  }
  return writeMap(map);
}

export function deletePerModelConfig(
  modelId: string,
  ggufVariant?: string | null,
): boolean {
  const map = readMap();
  // Mirror savePerModelConfig: never let an older client destroy a
  // future-schema entry it cannot interpret.
  if (hasFutureConfigForModelVariant(map, modelId, ggufVariant)) {
    return false;
  }
  if (!deleteConfigEntriesForModelVariant(map, modelId, ggufVariant)) {
    return true;
  }
  return writeMap(map);
}

export function resolveInitialConfig(
  modelId: string,
  ggufVariant?: string | null,
): { config: PerModelConfig; remembered: boolean } {
  const saved = loadPerModelConfig(modelId, ggufVariant);
  if (saved) {
    return { config: saved, remembered: true };
  }
  return { config: { ...DEFAULT_PER_MODEL_CONFIG }, remembered: false };
}
