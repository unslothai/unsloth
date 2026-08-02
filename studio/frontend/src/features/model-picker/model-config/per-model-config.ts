// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ggufVariantFromStorageKey,
  modelIdFromStorageKey,
  modelStorageKey,
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
  publicModelId,
} from "./model-identity";
import type { GpuIndexKind } from "@/hooks/use-gpu-info";

export interface PerModelConfig {
  customContextLength: number | null;
  maxSeqLength: number | null;
  kvCacheDtype: string | null;
  speculativeType: string | null;
  specDraftNMax: number | null;
  nParallel: number | null;
  tensorParallel: boolean;
  chatTemplateOverride: string | null;
  // GPU Memory controls (per-model, GGUF-only), optional so older blobs still
  // parse. null or absent selectedGpuIds means automatic placement; an array is
  // an explicit candidate pool. The --tensor-split ratio is deliberately not
  // remembered because it is bound to the exact GPU set and order.
  gpuMemoryMode?: "auto" | "manual";
  gpuLayers?: number;
  nCpuMoe?: number;
  selectedGpuIds?: number[] | null;
  selectedGpuIndexKind?: GpuIndexKind | null;
}

export const DEFAULT_PER_MODEL_CONFIG: PerModelConfig = {
  customContextLength: null,
  maxSeqLength: null,
  kvCacheDtype: null,
  speculativeType: null,
  specDraftNMax: null,
  nParallel: null,
  tensorParallel: false,
  chatTemplateOverride: null,
};

// Mirrors llama_server_args.py PARALLEL_MIN/MAX (LoadRequest.n_parallel
// bounds). null = follow the server-wide default.
export const N_PARALLEL_MIN = 1;
export const N_PARALLEL_MAX = 64;

export const MAX_SEQ_LENGTH_MIN = 128;
export const MAX_SEQ_LENGTH_MAX = 1048576;
export const MAX_SEQ_LENGTH_STEP = 128;
// App-default max sequence length when a non-GGUF model has no override. Both
// paths fall back to this rather than an active model's runtime value, so an
// unconfigured pane never inherits another model's larger context and OOMs.
export const DEFAULT_MAX_SEQ_LENGTH = 4096;
export const CONTEXT_LENGTH_MIN = 128;

// Matches studio/backend/core/inference/llama_cpp.py _valid_cache_types (f16 is the UI default).
export const KV_CACHE_DTYPES = [
  "bf16",
  "q8_0",
  "q4_0",
  "q4_1",
  "q5_0",
  "q5_1",
  "iq4_nl",
  "f32",
] as const;
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
  "nParallel",
  "tensorParallel",
  "chatTemplateOverride",
  "gpuMemoryMode",
  "gpuLayers",
  "nCpuMoe",
  "selectedGpuIds",
  "selectedGpuIndexKind",
]);

function normalizeGpuFields(partial: RawConfig): {
  gpuMemoryMode?: "auto" | "manual";
  gpuLayers?: number;
  nCpuMoe?: number;
  selectedGpuIds?: number[] | null;
  selectedGpuIndexKind?: GpuIndexKind | null;
} {
  const out: {
    gpuMemoryMode?: "auto" | "manual";
    gpuLayers?: number;
    nCpuMoe?: number;
    selectedGpuIds?: number[] | null;
    selectedGpuIndexKind?: GpuIndexKind | null;
  } = {};
  // Only "manual" is a real override; persisting "auto" would pin the model and
  // stop it following later changes to the global GPU Memory preference.
  if (partial.gpuMemoryMode === "manual") {
    out.gpuMemoryMode = "manual";
  }
  if (
    typeof partial.gpuLayers === "number" &&
    Number.isFinite(partial.gpuLayers)
  ) {
    out.gpuLayers = Math.trunc(partial.gpuLayers);
  }
  if (
    typeof partial.nCpuMoe === "number" &&
    Number.isFinite(partial.nCpuMoe) &&
    partial.nCpuMoe >= 0
  ) {
    out.nCpuMoe = Math.trunc(partial.nCpuMoe);
  }
  if (partial.selectedGpuIds === null) {
    out.selectedGpuIds = null;
  } else if (
    Array.isArray(partial.selectedGpuIds) &&
    partial.selectedGpuIds.every(
      (n) => typeof n === "number" && Number.isFinite(n),
    )
  ) {
    out.selectedGpuIds = partial.selectedGpuIds.map((n) => Math.trunc(n));
  }
  if (
    partial.selectedGpuIndexKind === "physical" ||
    partial.selectedGpuIndexKind === "vulkan" ||
    partial.selectedGpuIndexKind === null
  ) {
    out.selectedGpuIndexKind = partial.selectedGpuIndexKind;
  }
  return out;
}

function canonicalizeSpeculativeType(value: string): string | null {
  const s = value.trim().toLowerCase();
  if (!s) {
    return null;
  }
  // "auto"/"default" is the follow-global sentinel; store as null so it is never
  // persisted as an override and global speculative-decoding changes keep applying.
  if (s === "auto" || s === "default") {
    return null;
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

export function floorMaxSeqLength(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  const snapped = Math.floor(value / MAX_SEQ_LENGTH_STEP) * MAX_SEQ_LENGTH_STEP;
  return Math.max(MAX_SEQ_LENGTH_MIN, Math.min(MAX_SEQ_LENGTH_MAX, snapped));
}

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

// Whether Run Settings shows its advanced section is a standing preference,
// not a per-model one: opening it once keeps it open for every model and
// quant. Closed until asked for.
export const ADVANCED_SETTINGS_OPEN_KEY = "unsloth_model_advanced_settings";

function loadAdvancedSettingsOpen(): boolean | null {
  if (!canUseStorage()) {
    return null;
  }
  try {
    const raw = localStorage.getItem(ADVANCED_SETTINGS_OPEN_KEY);
    return raw === "true" ? true : raw === "false" ? false : null;
  } catch {
    return null;
  }
}

// Set only when a write is refused, so the switch keeps working in a browser
// with storage disabled, sandboxed or full. Cleared by the next write that
// sticks, which puts storage back in charge. `stored` is what storage held at
// the time, the one signal that tells a later write by someone else apart.
let unpersisted: { open: boolean; stored: boolean | null } | null = null;
const advancedOpenListeners = new Set<() => void>();

/** null until the switch is used, so an untouched panel is free to open the
 *  section for a model that carries non-default advanced values.
 *
 *  Read straight from storage rather than cached: a write from another tab
 *  while every panel was unmounted has no listener to catch it, and its
 *  storage event is not replayed on the next mount. */
export function readAdvancedSettingsOpen(): boolean | null {
  const stored = loadAdvancedSettingsOpen();
  if (!unpersisted) {
    return stored;
  }
  // Storage moved since the refused write, so someone made a newer choice and
  // it outranks the fallback. Checked on read, not on the storage event, so it
  // still holds for an event that landed while nothing was mounted.
  if (stored !== unpersisted.stored) {
    unpersisted = null;
    return stored;
  }
  return unpersisted.open;
}

/** True when the choice reached storage. */
function writeAdvancedSettingsOpen(open: boolean): boolean {
  if (!canUseStorage()) {
    return false;
  }
  try {
    localStorage.setItem(ADVANCED_SETTINGS_OPEN_KEY, open ? "true" : "false");
    return true;
  } catch {
    return false;
  }
}

export function saveAdvancedSettingsOpen(open: boolean): void {
  unpersisted = writeAdvancedSettingsOpen(open)
    ? null
    : { open, stored: loadAdvancedSettingsOpen() };
  // Run Settings is mounted on several surfaces at once, and the sidebar copy
  // stays mounted while collapsed, so tell them all rather than let them keep
  // a snapshot taken at mount.
  for (const listener of [...advancedOpenListeners]) {
    listener();
  }
}

/** Follow the preference while mounted, including a change from another tab. */
export function subscribeAdvancedSettingsOpen(
  onChange: () => void,
): () => void {
  advancedOpenListeners.add(onChange);
  if (!canUseStorage()) {
    return () => {
      advancedOpenListeners.delete(onChange);
    };
  }
  const onStorage = (event: StorageEvent) => {
    // A null key is a clear(), which drops this preference too.
    if (event.key === null || event.key === ADVANCED_SETTINGS_OPEN_KEY) {
      onChange();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    advancedOpenListeners.delete(onChange);
    window.removeEventListener("storage", onStorage);
  };
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
  evicted?: string[],
): { key: string; value: StoredMap[string] } | null {
  for (const key of Object.keys(map)) {
    // Never evict a future-schema entry an older client cannot interpret.
    if (
      protectedKeys?.has(key) ||
      storedConfigVersion(map[key]) > STORAGE_SCHEMA_VERSION
    ) {
      continue;
    }
    const value = map[key];
    delete map[key];
    evicted?.push(key);
    return { key, value };
  }
  return null;
}

function enforceStorageBudget(
  map: StoredMap,
  protectedKeys?: ReadonlySet<string>,
  evicted?: string[],
): boolean {
  let entryCount = Object.keys(map).length;
  while (entryCount > MAX_ENTRIES) {
    if (!deleteOldestEvictableEntry(map, protectedKeys, evicted)) {
      return false;
    }
    entryCount -= 1;
  }
  let bytes = serializedMapSize(map);
  while (bytes > MAX_PER_MODEL_CONFIG_STORAGE_BYTES) {
    const removed = deleteOldestEvictableEntry(map, protectedKeys, evicted);
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
    // Legacy blobs predate the parallel-slots knob.
    nParallel: null,
    tensorParallel:
      typeof raw.tensorParallel === "boolean" ? raw.tensorParallel : false,
    chatTemplateOverride: null,
    // Carry legacy GPU Memory knobs; normalizeGpuFields drops anything malformed.
    gpuMemoryMode:
      raw.gpuMemoryMode === "auto" || raw.gpuMemoryMode === "manual"
        ? raw.gpuMemoryMode
        : undefined,
    gpuLayers: typeof raw.gpuLayers === "number" ? raw.gpuLayers : undefined,
    nCpuMoe: typeof raw.nCpuMoe === "number" ? raw.nCpuMoe : undefined,
    selectedGpuIds:
      raw.selectedGpuIds === null
        ? null
        : Array.isArray(raw.selectedGpuIds)
          ? (raw.selectedGpuIds as number[])
          : undefined,
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
    // Snapshot existing entries so eviction can protect them: importing old load
    // settings must never discard a newer per-model config the user already has.
    const existingKeys = new Set(Object.keys(map));
    const migratedKeys = mergeLegacyEntries(
      map,
      legacy as Record<string, unknown>,
    );
    if (migratedKeys.length === 0) {
      localStorage.setItem(LEGACY_MIGRATION_FLAG, "1");
      return;
    }
    // Protect pre-existing entries so only just-migrated legacy entries are
    // dropped when over budget.
    if (!enforceStorageBudget(map, existingKeys)) {
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

function warnDroppedFields(
  raw: Record<string, unknown>,
  version: number,
): void {
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
  const speculativeType =
    rawSpecType ?? DEFAULT_PER_MODEL_CONFIG.speculativeType;
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
    nParallel:
      typeof partial.nParallel === "number" &&
      Number.isFinite(partial.nParallel)
        ? Math.max(
            N_PARALLEL_MIN,
            Math.min(N_PARALLEL_MAX, Math.round(partial.nParallel)),
          )
        : null,
    tensorParallel:
      typeof partial.tensorParallel === "boolean"
        ? partial.tensorParallel
        : DEFAULT_PER_MODEL_CONFIG.tensorParallel,
    chatTemplateOverride:
      typeof partial.chatTemplateOverride === "string" &&
      isChatTemplateWithinLimit(partial.chatTemplateOverride)
        ? partial.chatTemplateOverride
        : null,
    ...normalizeGpuFields(partial),
  };
}

/**
 * A config in the exact shape storage keeps it in: the UI carries sentinels storage does not
 * (Speculative Decoding "auto" canonicalizes to null), which would read as non-default.
 */
export function normalizePerModelConfig(raw: unknown): PerModelConfig {
  return normalize(raw);
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
  // Never apply a future-schema record an older client cannot interpret.
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
    config.nParallel == null &&
    Boolean(config.tensorParallel) ===
      Boolean(DEFAULT_PER_MODEL_CONFIG.tensorParallel) &&
    (config.chatTemplateOverride ?? null) === null &&
    gpuFieldsAtDefault(config)
  );
}

// GPU knobs are "default" when mode is Auto with no explicit choice: mode
// auto/absent, gpuLayers < 0/absent, nCpuMoe 0/absent, selectedGpuIds null/absent.
function gpuFieldsAtDefault(config: PerModelConfig): boolean {
  return (
    (config.gpuMemoryMode ?? "auto") === "auto" &&
    (config.gpuLayers == null || config.gpuLayers < 0) &&
    (config.nCpuMoe == null || config.nCpuMoe === 0) &&
    config.selectedGpuIds == null
  );
}

export function savePerModelConfig(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig,
  /**
   * Receives models dropped to stay inside the storage budget. Eviction is silent and
   * still reports success, so without this their server overrides would keep applying
   * with nothing in the UI able to forget them.
   */
  evicted?: { modelId: string; ggufVariant: string | null }[],
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
  const evictedKeys: string[] = [];
  if (!enforceStorageBudget(map, new Set([key]), evictedKeys)) {
    return false;
  }
  const written = writeMap(map);
  if (written && evicted) {
    for (const evictedKey of evictedKeys) {
      const id = modelIdFromStorageKey(evictedKey);
      if (!id) {
        continue;
      }
      const variant = ggufVariantFromStorageKey(evictedKey);
      evicted.push({ modelId: id, ggufVariant: variant ? variant : null });
    }
  }
  return written;
}

/** Every saved per-model config, decoded back to the ids it was keyed by. */
export function listPerModelConfigs(): {
  modelId: string;
  ggufVariant: string | null;
  config: PerModelConfig;
}[] {
  const out: {
    modelId: string;
    ggufVariant: string | null;
    config: PerModelConfig;
  }[] = [];
  for (const [key, raw] of Object.entries(readMap())) {
    const modelId = modelIdFromStorageKey(key);
    if (!modelId) {
      continue;
    }
    // Never report a future-schema record: loadPerModelConfig refuses to apply one, so
    // the backfill would persist this client's partial reading of it.
    if (storedConfigVersion(raw) > STORAGE_SCHEMA_VERSION) {
      continue;
    }
    const variant = ggufVariantFromStorageKey(key);
    out.push({
      modelId,
      ggufVariant: variant ? variant : null,
      config: normalize(raw),
    });
  }
  return out;
}

export function deletePerModelConfig(
  modelId: string,
  ggufVariant?: string | null,
): boolean {
  const map = readMap();
  // Mirror savePerModelConfig: never let an older client destroy a future-schema entry.
  if (hasFutureConfigForModelVariant(map, modelId, ggufVariant)) {
    return false;
  }
  if (!deleteConfigEntriesForModelVariant(map, modelId, ggufVariant)) {
    return true;
  }
  return writeMap(map);
}

/**
 * Move a saved config from an id an older release keyed it by onto the current one.
 *
 * A repo cached outside the active HF cache is now keyed by its repo id, because that is
 * what the picker and the auto-switch index use; it used to be keyed by the snapshot
 * path it loads from. Nothing else migrates that: the server backfill only mirrors what
 * is already stored, so without this the model reads as never remembered and comes up on
 * defaults after an upgrade.
 *
 * The key is renamed in one write rather than saved and then deleted: holding both copies at once
 * puts an already-full map over its budget, and the save then evicts the oldest unrelated model
 * silently, with no eviction list to hand back, so that model's server override outlives anything
 * the UI could forget. A rename cannot grow the entry count.
 */
export function adoptLegacyConfigKey(
  modelId: string,
  legacyModelId: string,
  ggufVariant?: string | null,
): boolean {
  if (!legacyModelId || legacyModelId === modelId) {
    return false;
  }
  const map = readMap();
  const legacyKey = findConfigKeyForModelVariant(
    map,
    legacyModelId,
    ggufVariant,
  );
  if (!legacyKey) {
    return false;
  }
  // Never interpret, move or destroy a record a newer client wrote, on either id: the same
  // guard loadPerModelConfig, savePerModelConfig and deletePerModelConfig each apply.
  if (
    storedConfigVersion(map[legacyKey]) > STORAGE_SCHEMA_VERSION ||
    hasFutureConfigForModelVariant(map, legacyModelId, ggufVariant) ||
    hasFutureConfigForModelVariant(map, modelId, ggufVariant)
  ) {
    return false;
  }
  const legacy = normalize(map[legacyKey]);
  // What is already saved under modelId wins; the stale record still goes.
  const alreadySaved =
    findConfigKeyForModelVariant(map, modelId, ggufVariant) !== null;
  const bytesBefore = serializedMapSize(map);
  delete map[legacyKey];
  deleteConfigEntriesForModelVariant(map, legacyModelId, ggufVariant);
  if (!(alreadySaved || isDefaultConfig(legacy))) {
    const [key] = storageKeysForModelVariant(modelId, ggufVariant);
    map[key] = toStoredConfig(legacy);
  }
  // Only the key strings change length, and a repo id is normally shorter than the snapshot
  // path it replaces. If it is not, and that tips the map past its byte cap, leave storage as
  // it was: the stale record is still readable, where an eviction would not be undoable.
  const bytesAfter = serializedMapSize(map);
  if (
    bytesAfter > bytesBefore &&
    bytesAfter > MAX_PER_MODEL_CONFIG_STORAGE_BYTES
  ) {
    return false;
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

/**
 * Remembered settings for the identifier ``/api/inference/status`` reports as loaded.
 *
 * An API auto-switch hands the loader the concrete snapshot path (the resolver index only holds
 * paths), so ``model_identifier`` names that path while this model's settings are keyed by its
 * repo id -- what ``modelConfigIdentity`` writes for a cached repo, and what the backend's
 * override lookup already tries alongside the path. Reading the raw identifier alone reports the
 * resident model as unremembered, blanking a control it is running with, which the next save then
 * writes back over the saved record. Only a namespaced collapse is adopted, the rule
 * ``residentModelIdMatches`` applies: an HF snapshot collapses onto a repo id naming exactly one
 * model, while every other path collapses onto a stem two models can share.
 */
export function resolveResidentInitialConfig(
  modelId: string,
  ggufVariant?: string | null,
): { config: PerModelConfig; remembered: boolean } {
  const direct = resolveInitialConfig(modelId, ggufVariant);
  if (direct.remembered) {
    return direct;
  }
  const alias = publicModelId(modelId);
  if (alias === modelId || !alias.includes("/")) {
    return direct;
  }
  return resolveInitialConfig(alias, ggufVariant);
}
