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
import { DRAFT_N_MAX_SPEC_TYPES } from "@/lib/speculative-modes";

export interface PerModelConfig {
  customContextLength: number | null;
  maxSeqLength: number | null;
  kvCacheDtype: string | null;
  /** MLX KV cache quantization width. Optional so older blobs still parse. */
  mlxKvBits?: number | null;
  speculativeType: string | null;
  specDraftNMax: number | null;
  nParallel: number | null;
  nBatch: number | null;
  nUbatch: number | null;
  tensorParallel: boolean;
  chatTemplateOverride: string | null;
  // GPU Memory controls (per-model, GGUF-only), optional so older blobs still parse. null/absent
  // selectedGpuIds means automatic. --tensor-split is not remembered: it is bound to the GPU set.
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
  mlxKvBits: null,
  speculativeType: null,
  specDraftNMax: null,
  nParallel: null,
  nBatch: null,
  nUbatch: null,
  tensorParallel: false,
  chatTemplateOverride: null,
};

// Mirrors llama_server_args.py PARALLEL_MIN/MAX; null = follow the server-wide default.
export const N_PARALLEL_MIN = 1;
export const N_PARALLEL_MAX = 64;

// Mirrors vram_budget_settings.py VRAM_FRACTION_MIN/MAX/DEFAULT as whole percent
// (the slider works in integers). Server-wide, so these are only the bounds the
// control clamps to; the value lives in /api/settings/vram-budget.
export const VRAM_BUDGET_PERCENT_MIN = 80;
export const VRAM_BUDGET_PERCENT_MAX = 100;
export const VRAM_BUDGET_PERCENT_DEFAULT = 97;
// Tenths: a whole percent is ~245 MiB of context on a 24 GB card. Mirrors
// VRAM_FRACTION_DECIMALS = 3 in vram_budget_settings.py.
export const VRAM_BUDGET_PERCENT_STEP = 0.1;

/** Percent for the slider; the fraction is rebuilt at the API boundary. */
export function vramFractionToPercent(fraction: number): number {
  return Math.round(fraction * 1000) / 10;
}

export function vramPercentToFraction(percent: number): number {
  // Three decimals, so 0.975 cannot come back as 0.9750000000000001 and read as
  // "changed" after a drag that ended where it began.
  return Math.round(percent * 10) / 1000;
}

// mirrors llama_server_args.py BATCH_MIN/MAX; null = follow the llama.cpp defaults (2048 / 512)
export const N_BATCH_MIN = 1;
export const N_BATCH_MAX = 65536;
// llama.cpp's own --batch-size default (_DEFAULT_LLAMA_N_BATCH), which a blank control
// runs at: it still caps the micro-batch, so advisories have to reckon with it.
export const N_BATCH_LLAMA_DEFAULT = 2048;

export const MAX_SEQ_LENGTH_MIN = 128;
export const MAX_SEQ_LENGTH_MAX = 1048576;
export const MAX_SEQ_LENGTH_STEP = 128;
// App-default max sequence length when a non-GGUF model has no override. Both paths fall back
// here rather than an active model's runtime value, so an unconfigured pane never OOMs.
export const DEFAULT_MAX_SEQ_LENGTH = 4096;
export const CONTEXT_LENGTH_MIN = 128;

// Reasons a Mac still cannot serve with MLX.
const NO_MLX_REASONS = new Set([
  "mlx_unavailable",
  "intel_mac",
  "detection_failed",
]);

/** Whether MLX will serve this model, and so whether MLX-only settings apply.
 *
 *  Every non-GGUF model loads through MLX on a working Mac stack, including plain
 *  safetensors repos, so `!isGguf` alone would show these controls to CUDA users.
 */
export function isServedByMlx(
  isGguf: boolean,
  deviceType: string | null | undefined,
  chatOnlyReason?: string | null,
): boolean {
  return (
    !isGguf &&
    deviceType === "mac" &&
    !NO_MLX_REASONS.has(chatOnlyReason ?? "")
  );
}

export function presetLoadSettingNames(
  isGguf: boolean,
  deviceType: string | null | undefined,
  chatOnlyReason?: string | null,
): string {
  if (isGguf) {
    return "context length, KV cache dtype, speculative decoding, GPU layers";
  }
  return isServedByMlx(isGguf, deviceType, chatOnlyReason)
    ? "max seq length, KV cache dtype"
    : "max seq length";
}

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

// Every width mx.quantize supports. By bit width, not a dtype name, hence separate
// from KV_CACHE_DTYPES.
export const MLX_KV_BITS: readonly number[] = [8, 6, 5, 4, 3, 2];
const VALID_KV_CACHE_DTYPES = new Set<string>(KV_CACHE_DTYPES);

export {
  DRAFT_N_MAX_SPEC_TYPES,
  SPECULATIVE_TYPES,
} from "@/lib/speculative-modes";

const STORAGE_KEY = "unsloth_model_configs";
const LEGACY_STORAGE_KEY = "unsloth_load_settings";
const LEGACY_MIGRATION_FLAG = "unsloth_model_configs_migrated";
// v2 added nBatch / nUbatch; a v1 client's normalizer would rewrite them away
const STORAGE_SCHEMA_VERSION = 2;
const PRE_BATCH_SCHEMA_VERSION = 1;
const MAX_ENTRIES = 500;
const MAX_PER_MODEL_CONFIG_STORAGE_BYTES = 1024 * 1024;
export const MAX_CHAT_TEMPLATE_BYTES = 65_536;

type StoredPerModelConfig = PerModelConfig & {
  version: number;
};
type StoredMap = Record<string, PerModelConfig | StoredPerModelConfig>;
type RawConfig = Partial<PerModelConfig> & { version?: unknown };

const STORED_CONFIG_FIELDS = new Set([
  "version",
  "customContextLength",
  "maxSeqLength",
  "kvCacheDtype",
  "mlxKvBits",
  "speculativeType",
  "specDraftNMax",
  "nParallel",
  "nBatch",
  "nUbatch",
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
  // Only "manual" is a real override; persisting "auto" would stop the model following the global.
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
  // "auto"/"default" is the follow-global sentinel; store as null so it is never an override.
  if (s === "auto" || s === "default") {
    return null;
  }
  if (s === "off") {
    return "off";
  }
  if (s === "mtp" || s === "draft-mtp") {
    return "mtp";
  }
  if (s === "dspark" || s === "draft-dspark") {
    return "dspark";
  }
  if (s === "dflash" || s === "draft-dflash") {
    return "dflash";
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

// Whether Run Settings shows its advanced section is a standing preference, not per-model:
// opening it once keeps it open for every model and quant. Closed until asked for.
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

// Set only when a write is refused, so the switch keeps working in a browser with storage
// disabled or full. Cleared by the next write that sticks. `stored` is what storage held at
// the time, the one signal that tells a later write by someone else apart.
let unpersisted: { open: boolean; stored: boolean | null } | null = null;
const advancedOpenListeners = new Set<() => void>();

/** null until the switch is used, so an untouched panel is free to open the
*  section for a model that carries non-default advanced values.
*  Read straight from storage rather than cached: a write from another tab while every panel
*  was unmounted has no listener to catch it, and its storage event is not replayed on mount. */
export function readAdvancedSettingsOpen(): boolean | null {
  const stored = loadAdvancedSettingsOpen();
  if (!unpersisted) {
    return stored;
  }
  // Storage moved since the refused write, so a newer choice outranks the fallback. Checked on
  // read, not on the storage event, so it still holds for an event that landed while unmounted.
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
  // Run Settings is mounted on several surfaces at once, and the sidebar copy stays mounted
  // while collapsed, so tell them all rather than let them keep a snapshot taken at mount.
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
    // Snapshot existing entries so eviction protects them: importing old load settings must never discard a newer config.
    const existingKeys = new Set(Object.keys(map));
    const migratedKeys = mergeLegacyEntries(
      map,
      legacy as Record<string, unknown>,
    );
    if (migratedKeys.length === 0) {
      localStorage.setItem(LEGACY_MIGRATION_FLAG, "1");
      return;
    }
    // Protect pre-existing entries so only just-migrated legacy entries are dropped over budget.
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
    DRAFT_N_MAX_SPEC_TYPES.has(speculativeType) &&
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
    mlxKvBits:
      typeof partial.mlxKvBits === "number" &&
      MLX_KV_BITS.includes(partial.mlxKvBits)
        ? partial.mlxKvBits
        : null,
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
    nBatch:
      typeof partial.nBatch === "number" && Number.isFinite(partial.nBatch)
        ? Math.max(N_BATCH_MIN, Math.min(N_BATCH_MAX, Math.round(partial.nBatch)))
        : null,
    nUbatch:
      typeof partial.nUbatch === "number" && Number.isFinite(partial.nUbatch)
        ? Math.max(N_BATCH_MIN, Math.min(N_BATCH_MAX, Math.round(partial.nUbatch)))
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
* (Speculative Decoding "auto" canonicalizes to null), which would read as non-default. */
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
  const normalized = normalize(config);
  // records without the v2-only batch fields keep v1 so older clients can still rewrite them
  const version =
    normalized.nBatch != null || normalized.nUbatch != null
      ? STORAGE_SCHEMA_VERSION
      : PRE_BATCH_SCHEMA_VERSION;
  return {
    version,
    ...normalized,
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
    (config.mlxKvBits ?? null) === DEFAULT_PER_MODEL_CONFIG.mlxKvBits &&
    config.speculativeType === DEFAULT_PER_MODEL_CONFIG.speculativeType &&
    config.specDraftNMax == null &&
    config.nParallel == null &&
    config.nBatch == null &&
    config.nUbatch == null &&
    Boolean(config.tensorParallel) ===
      Boolean(DEFAULT_PER_MODEL_CONFIG.tensorParallel) &&
    (config.chatTemplateOverride ?? null) === null &&
    gpuFieldsAtDefault(config)
  );
}

// GPU knobs are "default" when mode is Auto with no explicit choice: gpuLayers < 0/absent, nCpuMoe 0/absent, selectedGpuIds null/absent.
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
  * Receives models dropped to stay inside the storage budget. Eviction is silent and still
  * reports success, so without this their server overrides would keep applying with nothing
  * in the UI able to forget them. */
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
    // Never report a future-schema record: loadPerModelConfig refuses to apply one anyway.
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
* A repo cached outside the active HF cache is now keyed by its repo id (what the picker and
* auto-switch index use); it used to be keyed by the snapshot path it loads from. Nothing else
* migrates that, so without this the model reads as never remembered after an upgrade.
*
* The key is renamed in one write rather than saved then deleted: holding both copies puts an
* already-full map over budget, and the save then silently evicts an unrelated model whose
* server override outlives anything the UI could forget. A rename cannot grow the entry count.
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
  // Never interpret, move or destroy a record a newer client wrote, on either id.
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
  // Only the key strings change length, and a repo id is normally shorter than the snapshot path.
  // If it is not and that tips the map past its byte cap, leave storage as it was: eviction is not undoable.
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
* repo id. Reading the raw identifier alone reports the resident model as unremembered, blanking
* a control it is running with, which the next save writes back over the saved record. Only a
* namespaced collapse is adopted, per ``residentModelIdMatches``: an HF snapshot collapses onto
* a repo id naming exactly one model, while other paths collapse onto a shareable stem. */
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
