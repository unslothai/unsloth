// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface PerModelConfig {
  kvCacheDtype: string | null;
  speculativeType: string | null;
  specDraftNMax: number | null;
  customContextLength: number | null;
  chatTemplateOverride: string | null;
  trustRemoteCode?: boolean;
}

export const DEFAULT_PER_MODEL_CONFIG: PerModelConfig = {
  kvCacheDtype: null,
  speculativeType: "auto",
  specDraftNMax: null,
  customContextLength: null,
  chatTemplateOverride: null,
  trustRemoteCode: false,
};

// Stored KV cache dtypes (full precision = null, rendered as "f16" in the UI).
export const KV_CACHE_DTYPES = ["bf16", "q8_0", "q5_1", "q4_1"] as const;
const VALID_KV_CACHE_DTYPES = new Set<string>(KV_CACHE_DTYPES);
// Single source of truth for the speculative-decoding modes. UI labels live in
// model-config-page; the values and the MTP-bearing subset (which gates
// specDraftNMax) are shared from here so the two can't drift.
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
const VALID_SPECULATIVE_TYPES = new Set<string>(SPECULATIVE_TYPES);

const STORAGE_KEY = "unsloth_model_configs";
const MAX_ENTRIES = 500;
export const MAX_CHAT_TEMPLATE_LENGTH = 65_536;

type StoredMap = Record<string, PerModelConfig>;

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

export function configEntryKey(
  modelId: string,
  ggufVariant?: string | null,
): string {
  return `${modelId}::${ggufVariant ?? ""}`;
}

function readMap(): StoredMap {
  if (!canUseStorage()) return {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    return parsed as StoredMap;
  } catch {
    return {};
  }
}

function writeMap(map: StoredMap): boolean {
  if (!canUseStorage()) return false;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
    return true;
  } catch (err) {
    console.warn("Failed to persist per-model config:", err);
    return false;
  }
}

function normalize(raw: unknown): PerModelConfig {
  const partial = (raw && typeof raw === "object" ? raw : {}) as Partial<PerModelConfig>;
  const rawSpecType =
    typeof partial.speculativeType === "string"
      ? partial.speculativeType === "default"
        ? "auto"
        : partial.speculativeType
      : null;
  const speculativeType =
    rawSpecType != null && VALID_SPECULATIVE_TYPES.has(rawSpecType)
      ? rawSpecType
      : DEFAULT_PER_MODEL_CONFIG.speculativeType;
  const specDraftNMax =
    speculativeType != null &&
    MTP_SPECULATIVE_TYPES.has(speculativeType) &&
    typeof partial.specDraftNMax === "number" &&
    Number.isFinite(partial.specDraftNMax)
      ? Math.max(1, Math.min(16, Math.round(partial.specDraftNMax)))
      : null;
  return {
    kvCacheDtype:
      typeof partial.kvCacheDtype === "string" &&
      VALID_KV_CACHE_DTYPES.has(partial.kvCacheDtype)
        ? partial.kvCacheDtype
        : null,
    speculativeType,
    specDraftNMax,
    customContextLength:
      typeof partial.customContextLength === "number" &&
      Number.isFinite(partial.customContextLength) &&
      partial.customContextLength > 0
        ? Math.floor(partial.customContextLength)
        : null,
    chatTemplateOverride:
      typeof partial.chatTemplateOverride === "string" &&
      partial.chatTemplateOverride.length <= MAX_CHAT_TEMPLATE_LENGTH
        ? partial.chatTemplateOverride
        : null,
    trustRemoteCode:
      typeof partial.trustRemoteCode === "boolean"
        ? partial.trustRemoteCode
        : DEFAULT_PER_MODEL_CONFIG.trustRemoteCode,
  };
}

export function loadPerModelConfig(
  modelId: string,
  ggufVariant?: string | null,
): PerModelConfig | null {
  const key = configEntryKey(modelId, ggufVariant);
  const map = readMap();
  if (!Object.hasOwn(map, key)) return null;
  return normalize(map[key]);
}

export function hasPerModelConfig(
  modelId: string,
  ggufVariant?: string | null,
): boolean {
  return loadPerModelConfig(modelId, ggufVariant) != null;
}

export function isDefaultConfig(config: PerModelConfig): boolean {
  return (
    config.customContextLength == null &&
    (config.kvCacheDtype ?? null) === DEFAULT_PER_MODEL_CONFIG.kvCacheDtype &&
    config.speculativeType === DEFAULT_PER_MODEL_CONFIG.speculativeType &&
    config.specDraftNMax == null &&
    (config.chatTemplateOverride ?? null) === null &&
    Boolean(config.trustRemoteCode) ===
      Boolean(DEFAULT_PER_MODEL_CONFIG.trustRemoteCode)
  );
}

export function savePerModelConfig(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig,
): boolean {
  const normalized = normalize(config);
  if (isDefaultConfig(normalized)) {
    const map = readMap();
    const key = configEntryKey(modelId, ggufVariant);
    if (!Object.hasOwn(map, key)) return true;
    delete map[key];
    return writeMap(map);
  }
  const map = readMap();
  const key = configEntryKey(modelId, ggufVariant);
  delete map[key];
  map[key] = normalized;
  const keys = Object.keys(map);
  if (keys.length > MAX_ENTRIES) {
    for (const stale of keys.slice(0, keys.length - MAX_ENTRIES)) {
      delete map[stale];
    }
  }
  return writeMap(map);
}

export function deletePerModelConfig(
  modelId: string,
  ggufVariant?: string | null,
): void {
  const map = readMap();
  const key = configEntryKey(modelId, ggufVariant);
  if (!Object.hasOwn(map, key)) return;
  delete map[key];
  writeMap(map);
}

export function deletePerModelConfigsForModel(modelId: string): void {
  const map = readMap();
  const prefix = `${modelId}::`;
  let changed = false;
  for (const key of Object.keys(map)) {
    if (key.startsWith(prefix)) {
      delete map[key];
      changed = true;
    }
  }
  if (changed) writeMap(map);
}

export function resolveInitialConfig(
  modelId: string,
  ggufVariant?: string | null,
): { config: PerModelConfig; remembered: boolean } {
  const saved = loadPerModelConfig(modelId, ggufVariant);
  if (saved) return { config: saved, remembered: true };
  return { config: { ...DEFAULT_PER_MODEL_CONFIG }, remembered: false };
}
