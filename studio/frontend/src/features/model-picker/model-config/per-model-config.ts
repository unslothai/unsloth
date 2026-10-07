// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GpuIndexKind } from "@/hooks/use-gpu-info";
import {
  cachedRepoConfigId,
  ggufVariantFromStorageKey,
  isStandaloneGgufPath,
  modelIdFromStorageKey,
  modelStorageKey,
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
  publicModelId,
} from "./model-identity";
import { isExternalModelId } from "@/features/chat/external-providers";
import {
  DRAFT_N_MAX_SPEC_TYPES,
  SEPARATE_DRAFT_MODEL_SPEC_TYPES,
} from "@/lib/speculative-modes";

export interface PerModelConfig {
  engineParallelism?: "tensor" | "pipeline" | "data";
  enginePrecision?: "auto" | "bf16" | "fp16" | "int4" | "int8" | "fp8";
  engine?: "auto" | "vllm" | "sglang";
  customContextLength: number | null;
  maxSeqLength: number | null;
  kvCacheDtype: string | null;
  mlxKvQuant?: MlxKvQuant | null;
  mlxInt8Prefill?: boolean;
  speculativeType: string | null;
  specDraftNMax: number | null;
  /** Draft-context KV dtype, independent of kvCacheDtype. Optional so older blobs parse. */
  specDraftCacheDtype?: string | null;
  nParallel: number | null;
  reasoningBudget: number;
  reasoningBudgetMessage: string;
  nBatch: number | null;
  nUbatch: number | null;
  /** --load-mode; null lets the fit decide. */
  loadMode?: string | null;
  /** --ctx-checkpoints; null follows the llama.cpp default (32). */
  ctxCheckpoints?: number | null;
  /** --cache-ram in MiB; null follows the llama.cpp default (8192). */
  cacheRam?: number | null;
  tensorParallel: boolean;
  disableVision: boolean;
  chatTemplateOverride: string | null;
  /** Appended after Unsloth's flags. `undefined`: never read, so a save keeps the server's (CLI-set)
    flags; `null`: cleared, sent as `[]`; a list: launch with it. */
  llamaExtraArgs?: string[] | null;
  // Optional so older blobs parse; absent or null selectedGpuIds means automatic.
  gpuMemoryMode?: "auto" | "manual";
  gpuLayers?: number;
  nCpuMoe?: number;
  selectedGpuIds?: number[] | null;
  selectedGpuIndexKind?: GpuIndexKind | null;
  /** Never stored. `undefined` defers to the store, `null` = default. */
  tensorSplit?: number[] | null;
}

export const DEFAULT_PER_MODEL_CONFIG: PerModelConfig = {
  engine: "auto",
  enginePrecision: "auto",
  engineParallelism: "tensor",
  customContextLength: null,
  maxSeqLength: null,
  kvCacheDtype: null,
  mlxKvQuant: null,
  mlxInt8Prefill: false,
  speculativeType: null,
  specDraftNMax: null,
  specDraftCacheDtype: null,
  nParallel: null,
  reasoningBudget: -1,
  reasoningBudgetMessage: "",
  nBatch: null,
  nUbatch: null,
  loadMode: null,
  ctxCheckpoints: null,
  cacheRam: null,
  tensorParallel: false,
  disableVision: false,
  chatTemplateOverride: null,
};

// Mirrors llama_server_args.py PARALLEL_MIN/MAX; null = follow the server-wide default.
export const N_PARALLEL_MIN = 1;
export const N_PARALLEL_MAX = 64;

// Mirrors vram_budget_settings.py VRAM_FRACTION_MIN/MAX/DEFAULT as percent; value lives server-side.
export const VRAM_BUDGET_PERCENT_MIN = 80;
export const VRAM_BUDGET_PERCENT_MAX = 100;
export const VRAM_BUDGET_PERCENT_DEFAULT = 97;
// Tenths: a whole percent is ~245 MiB on a 24 GB card. Mirrors VRAM_FRACTION_DECIMALS = 3.
export const VRAM_BUDGET_PERCENT_STEP = 0.1;

export function vramFractionToPercent(fraction: number): number {
  return Math.round(fraction * 1000) / 10;
}

export function vramPercentToFraction(percent: number): number {
  // Three decimals, so a drag ending where it began does not read as changed.
  return Math.round(percent * 10) / 1000;
}

// Mirrors llama_server_args.py BATCH_MIN/MAX; null = llama.cpp defaults (2048 / 512).
export const N_BATCH_MIN = 1;
export const N_BATCH_MAX = 65536;
// A blank control still runs at this, so advisories must reckon with it.
export const N_BATCH_LLAMA_DEFAULT = 2048;

export const MAX_SEQ_LENGTH_MIN = 128;
export const MAX_SEQ_LENGTH_MAX = 1048576;
export const MAX_SEQ_LENGTH_STEP = 128;
// Never fall back to an active model's runtime value, or an unconfigured pane inherits and OOMs.
export const DEFAULT_MAX_SEQ_LENGTH = 4096;
export const CONTEXT_LENGTH_MIN = 128;

const NO_MLX_REASONS = new Set([
  "mlx_unavailable",
  "no_torch",
  "intel_mac",
  "detection_failed",
]);

/** `!isGguf` alone would show MLX controls to CUDA users. */
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

/** Native-audio checkpoints load on Apple Silicon without MLX; `loadedIsMlx` null means not loaded. */
export function residentIsServedByMlx(
  isGguf: boolean,
  deviceType: string | null | undefined,
  chatOnlyReason: string | null | undefined,
  loadedIsMlx: boolean | null | undefined,
): boolean {
  return (
    isServedByMlx(isGguf, deviceType, chatOnlyReason) && loadedIsMlx !== false
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

/** A context length is not evidence (MLX reports one); external providers keep a `.gguf` suffix. */
export function isServedByLlamaCpp(x: {
  loadedIsGguf?: boolean | null;
  activeGgufVariant?: string | null;
  activeNativePathToken?: string | null;
  checkpoint?: string | null;
}): boolean {
  if (isExternalModelId(x.checkpoint)) return false;
  // Variant and path token outlive the pick, so a reported non-GGUF backend settles it.
  if (x.loadedIsGguf === false) return false;
  return (
    x.loadedIsGguf === true ||
    x.activeGgufVariant != null ||
    x.activeNativePathToken != null ||
    String(x.checkpoint ?? "").toLowerCase().endsWith(".gguf")
  );
}

export function resumesThought(x: {
  loadedIsGguf?: boolean | null;
  loadedIsMlx?: boolean | null;
  activeGgufVariant?: string | null;
  activeNativePathToken?: string | null;
  checkpoint?: string | null;
}): boolean {
  return (
    isServedByLlamaCpp(x) ||
    (!isExternalModelId(x.checkpoint) && x.loadedIsMlx === true)
  );
}

/** MLX always sizes a window; transformers echoes max_seq_length, so without a native length it
  contributes none. The four fields move together. */
export function loadedContextFields(resp: {
  is_gguf?: boolean;
  is_mlx?: boolean;
  context_length?: number | null;
  native_context_length?: number | null;
  max_context_length?: number | null;
  context_length_enforced?: boolean | null;
  context_unbounded_when_batched?: boolean;
  parallel_slots?: number | null;
  mlx_context_budget?: number | null;
} | null): {
  loadedContextLength: number | null;
  maxContextLength: number | null;
  nativeContextLength: number | null;
  loadedIsGguf: boolean | null;
  loadedIsMlx: boolean | null;
  loadedContextEnforced: boolean | null;
  loadedContextUnboundedWhenBatched: boolean;
  loadedParallelSlots: number | null;
  loadedContextBudget: number | null;
} {
  if (!resp) {
    return {
      loadedContextLength: null,
      maxContextLength: null,
      nativeContextLength: null,
      loadedIsGguf: null,
      loadedIsMlx: null,
      loadedContextEnforced: null,
      loadedContextUnboundedWhenBatched: false,
      loadedParallelSlots: null,
      loadedContextBudget: null,
    };
  }
  const isGguf = resp.is_gguf ?? false;
  // Unknown, not a default: omitted when reading the window failed.
  const loaded = resp.context_length ?? null;
  if (!isGguf && !resp.is_mlx && resp.native_context_length == null) {
    return {
      loadedContextLength: null,
      maxContextLength: null,
      nativeContextLength: null,
      loadedIsGguf: false,
      loadedIsMlx: resp.is_mlx ?? null,
      loadedContextEnforced: null,
      loadedContextUnboundedWhenBatched: false,
      loadedParallelSlots: null,
      loadedContextBudget: null,
    };
  }
  return {
    loadedContextLength: loaded,
    maxContextLength: resp.max_context_length ?? loaded,
    nativeContextLength: resp.native_context_length ?? null,
    loadedIsGguf: isGguf,
    // Backend's answer, so native audio off the MLX path is not taken for MLX.
    loadedIsMlx: resp.is_mlx ?? null,
    // llama.cpp allocates what it reports, so GGUF is enforced by construction.
    loadedContextEnforced: isGguf ? true : (resp.context_length_enforced ?? null),
    // Same response as the other two, so they never mix across loads.
    loadedContextUnboundedWhenBatched: isGguf
      ? false
      : (resp.context_unbounded_when_batched ?? false),
    loadedParallelSlots: resp.parallel_slots ?? null,
    loadedContextBudget: isGguf ? null : (resp.mlx_context_budget ?? null),
  };
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

/** CUDA, ROCm and Metal lack an iq4_nl FlashAttention kernel. A selected value stays listed. */
export function kvCacheDtypeOptions(
  backend: string | null,
  selected: string | null | undefined,
): readonly string[] {
  if (backend === "vulkan" || backend === "cpu") return KV_CACHE_DTYPES;
  return KV_CACHE_DTYPES.filter((dtype) => dtype !== "iq4_nl" || dtype === selected);
}

export const MLX_KV_QUANTS = [
  "8",
  "6",
  "5",
  "4",
  "3",
  "2",
  "tq-4",
  "tq-3.5",
  "tq-3",
  "tq-2",
] as const;
export type MlxKvQuant = (typeof MLX_KV_QUANTS)[number];
const VALID_MLX_KV_QUANTS = new Set<string>(MLX_KV_QUANTS);

export function mlxKvQuantLabel(quant: string): string {
  return quant.startsWith("tq-") ? `TurboQuant ${quant.slice(3)}-bit` : `${quant}-bit`;
}

export function normalizeMlxKvQuant(
  value: unknown,
  supersededBits?: unknown,
): MlxKvQuant | null {
  if (typeof value === "string") {
    // Matches the backend's reader, or one row means two settings.
    const named = value.trim().toLowerCase();
    return VALID_MLX_KV_QUANTS.has(named) ? (named as MlxKvQuant) : null;
  }
  if (value !== undefined) return null;
  if (typeof supersededBits !== "number" || !Number.isFinite(supersededBits)) return null;
  const name = String(supersededBits);
  return VALID_MLX_KV_QUANTS.has(name) ? (name as MlxKvQuant) : null;
}
const VALID_KV_CACHE_DTYPES = new Set<string>(KV_CACHE_DTYPES);

// --load-mode enum in --help order. "auto" is stored as null and never sent: builds reject it.
export const LOAD_MODES = [
  "auto",
  "none",
  "mmap",
  "mlock",
  "mmap+mlock",
  "dio",
] as const;
export const LOAD_MODE_DEFAULT = "auto";
const VALID_LOAD_MODES = new Set<string>(LOAD_MODES);

// 0 disables; the ceiling is a sanity bound, not an upstream one.
export const CTX_CHECKPOINTS_MIN = 0;
export const CTX_CHECKPOINTS_MAX = 256;
export const CTX_CHECKPOINTS_LLAMA_DEFAULT = 32;

// -1 is "no limit" and 0 disables; the 1 TiB ceiling catches stray keystrokes.
export const CACHE_RAM_MIN = -1;
export const CACHE_RAM_MAX = 1024 * 1024;
export const CACHE_RAM_LLAMA_DEFAULT = 8192;

export {
  DRAFT_N_MAX_SPEC_TYPES,
  SEPARATE_DRAFT_MODEL_SPEC_TYPES,
  SPECULATIVE_TYPES,
} from "@/lib/speculative-modes";

export const PER_MODEL_CONFIG_STORAGE_KEY = "unsloth_model_configs";
const STORAGE_KEY = PER_MODEL_CONFIG_STORAGE_KEY;
const LEGACY_STORAGE_KEY = "unsloth_load_settings";
const LEGACY_MIGRATION_FLAG = "unsloth_model_configs_migrated";
// v2 nBatch/nUbatch, v3 llamaExtraArgs, v4 disableVision, v5 tuning group, v6 reasoning pair,
// v7 mlxKvQuant, v9 mlxInt8Prefill. v8 is skipped: nightly builds stamped it for a reverted config, so a v8 client must not
// claim to understand an int8 prefill record.
const STORAGE_SCHEMA_VERSION = 9;
const PRE_MLX_INT8_PREFILL_SCHEMA_VERSION = 7;
const PRE_MLX_KV_QUANT_SCHEMA_VERSION = 6;
const PRE_REASONING_BUDGET_SCHEMA_VERSION = 5;
const PRE_SERVER_TUNING_SCHEMA_VERSION = 4;
const PRE_VISION_SCHEMA_VERSION = 3;
const PRE_EXTRA_ARGS_SCHEMA_VERSION = 2;
const PRE_BATCH_SCHEMA_VERSION = 1;
const MAX_ENTRIES = 500;
const MAX_PER_MODEL_CONFIG_STORAGE_BYTES = 1024 * 1024;
export const MAX_CHAT_TEMPLATE_BYTES = 65_536;
export const MAX_REASONING_BUDGET_MESSAGE_BYTES = 8_192;

export function isReasoningBudgetMessageValid(value: string): boolean {
  if (value.includes("\0")) return false;
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(i + 1);
      if (next < 0xdc00 || next > 0xdfff) return false;
      i += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      return false;
    }
  }
  return (
    new TextEncoder().encode(value).byteLength <=
    MAX_REASONING_BUDGET_MESSAGE_BYTES
  );
}

type StoredPerModelConfig = PerModelConfig & {
  version: number;
};
type StoredMap = Record<string, PerModelConfig | StoredPerModelConfig>;
type RawConfig = Partial<PerModelConfig> & { version?: unknown; mlxKvBits?: unknown };

const STORED_CONFIG_FIELDS = new Set([
  "version",
  "customContextLength",
  "maxSeqLength",
  "kvCacheDtype",
  "mlxKvQuant",
  "mlxInt8Prefill",
  "speculativeType",
  "specDraftNMax",
  "specDraftCacheDtype",
  "nParallel",
  "reasoningBudget",
  "reasoningBudgetMessage",
  "nBatch",
  "nUbatch",
  "loadMode",
  "ctxCheckpoints",
  "cacheRam",
  "tensorParallel",
  "disableVision",
  "chatTemplateOverride",
  "llamaExtraArgs",
  "gpuMemoryMode",
  "gpuLayers",
  "nCpuMoe",
  "selectedGpuIds",
  "selectedGpuIndexKind",
]);

/** Anything not an array is "not loaded" (`undefined`), never "cleared". */
function normalizeLlamaExtraArgs(value: unknown): string[] | null | undefined {
  if (value === null) {
    return null;
  }
  if (!Array.isArray(value)) {
    return undefined;
  }
  const tokens = value.filter((entry): entry is string => typeof entry === "string");
  return tokens.length > 0 ? tokens : null;
}

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
  // Persisting "auto" would stop the model following the global.
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
  if (s === "auto" || s === "default") {
    return null;
  }
  // _LEGACY_SPEC_MODE_MAP's off spellings; null would mean follow-global and enable a drafter under Auto.
  if (s === "off" || s === "none" || s === "disable" || s === "disabled") {
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

/** "auto" folds to null: it is the default, which a build may redefine. */
export function canonicalizeLoadMode(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  // Whitespace and case only; unaccepted spellings are refused, not repaired.
  const mode = value.trim().toLowerCase();
  if (!mode || mode === LOAD_MODE_DEFAULT) {
    return null;
  }
  return VALID_LOAD_MODES.has(mode) ? mode : null;
}

function normalizeIntegerInRange(
  value: unknown,
  min: number,
  max: number,
): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return Math.max(min, Math.min(max, Math.round(value)));
}

export function normalizeCtxCheckpoints(value: unknown): number | null {
  return normalizeIntegerInRange(value, CTX_CHECKPOINTS_MIN, CTX_CHECKPOINTS_MAX);
}

export function normalizeCacheRam(value: unknown): number | null {
  return normalizeIntegerInRange(value, CACHE_RAM_MIN, CACHE_RAM_MAX);
}

export function normalizeMaxSeqLength(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  const snapped = Math.round(value / MAX_SEQ_LENGTH_STEP) * MAX_SEQ_LENGTH_STEP;
  return Math.max(MAX_SEQ_LENGTH_MIN, Math.min(MAX_SEQ_LENGTH_MAX, snapped));
}

/** Saved records only: MLX pins moved to `customContextLength`, older ones in `maxSeqLength`.
  For the live store read `customContextLength` alone. */
export function savedContextPin(config: {
  customContextLength?: number | null;
  maxSeqLength?: number | null;
}): number | null {
  return (
    config.customContextLength ??
    normalizeMaxSeqLength(config.maxSeqLength ?? null)
  );
}

/** Leaves the pin in exactly one field, since picker and API auto-switch prefer different ones. */
export function contextPinPatch(value: number, isMlx: boolean): Partial<PerModelConfig> {
  // Bounded but not rounded to the control's step.
  const pin = boundContextPin(value) ?? MAX_SEQ_LENGTH_MIN;
  return isMlx
    ? { customContextLength: pin, maxSeqLength: null }
    : { customContextLength: null, maxSeqLength: pin };
}

function boundContextPin(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  return Math.max(
    MAX_SEQ_LENGTH_MIN,
    Math.min(MAX_SEQ_LENGTH_MAX, Math.floor(value)),
  );
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

// Standing preference, not per-model; closed until asked for.
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

// Set only when a write is refused; `stored` detects a later write by someone else.
let unpersisted: { open: boolean; stored: boolean | null } | null = null;
const advancedOpenListeners = new Set<() => void>();

/** Read from storage, not cached: another tab may write while every panel is unmounted. */
export function readAdvancedSettingsOpen(): boolean | null {
  const stored = loadAdvancedSettingsOpen();
  if (!unpersisted) {
    return stored;
  }
  // Storage moved since the refused write, so a newer choice outranks the fallback.
  if (stored !== unpersisted.stored) {
    unpersisted = null;
    return stored;
  }
  return unpersisted.open;
}

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
  // Several mounted copies (sidebar stays mounted), so notify all.
  for (const listener of [...advancedOpenListeners]) {
    listener();
  }
}

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
    nParallel: null,
    reasoningBudget: -1,
    reasoningBudgetMessage: "",
    tensorParallel:
      typeof raw.tensorParallel === "boolean" ? raw.tensorParallel : false,
    disableVision:
      typeof raw.disableVision === "boolean" ? raw.disableVision : false,
    chatTemplateOverride: null,
    // Absent, not null: the server may hold CLI-set flags a "cleared" save would wipe.
    llamaExtraArgs: undefined,
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
    // Importing old load settings must never evict a newer config.
    const existingKeys = new Set(Object.keys(map));
    const migratedKeys = mergeLegacyEntries(
      map,
      legacy as Record<string, unknown>,
    );
    if (migratedKeys.length === 0) {
      localStorage.setItem(LEGACY_MIGRATION_FLAG, "1");
      return;
    }
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

/** Same-tab notification; the `storage` event only reaches other tabs. */
export const PER_MODEL_CONFIG_UPDATED_EVENT =
  "unsloth-per-model-config-updated";

function writeMap(map: StoredMap): boolean {
  if (!canUseStorage()) {
    return false;
  }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch (err) {
    console.warn("Failed to persist per-model config:", err);
    return false;
  }
  // Best-effort: the write already landed.
  if (typeof window?.dispatchEvent === "function") {
    window.dispatchEvent(new Event(PER_MODEL_CONFIG_UPDATED_EVENT));
  }
  return true;
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
  // A dtype stored under a mode with no separate drafter would show a phantom row.
  const specDraftCacheDtype =
    speculativeType != null &&
    SEPARATE_DRAFT_MODEL_SPEC_TYPES.has(speculativeType) &&
    typeof partial.specDraftCacheDtype === "string" &&
    VALID_KV_CACHE_DTYPES.has(partial.specDraftCacheDtype)
      ? partial.specDraftCacheDtype
      : null;
  return {
    engineParallelism: partial.engineParallelism === "pipeline" || partial.engineParallelism === "data"
      ? partial.engineParallelism : "tensor",
    enginePrecision: ["bf16", "fp16", "int4", "int8", "fp8"].includes(
      partial.enginePrecision ?? "",
    )
      ? partial.enginePrecision
      : "auto",
    engine:
      partial.engine === "vllm" || partial.engine === "sglang"
        ? partial.engine
        : "auto",
    customContextLength:
      typeof partial.customContextLength === "number" &&
      Number.isFinite(partial.customContextLength) &&
      partial.customContextLength > 0
        ? Math.max(CONTEXT_LENGTH_MIN, Math.floor(partial.customContextLength))
        : null,
    maxSeqLength: normalizeMaxSeqLength(partial.maxSeqLength),
    mlxKvQuant: normalizeMlxKvQuant(partial.mlxKvQuant, partial.mlxKvBits),
    mlxInt8Prefill: partial.mlxInt8Prefill === true,
    kvCacheDtype:
      typeof partial.kvCacheDtype === "string" &&
      VALID_KV_CACHE_DTYPES.has(partial.kvCacheDtype)
        ? partial.kvCacheDtype
        : null,
    speculativeType,
    specDraftNMax,
    specDraftCacheDtype,
    loadMode: canonicalizeLoadMode(partial.loadMode),
    ctxCheckpoints: normalizeCtxCheckpoints(partial.ctxCheckpoints),
    cacheRam: normalizeCacheRam(partial.cacheRam),
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
    reasoningBudget:
      typeof partial.reasoningBudget === "number" &&
      Number.isFinite(partial.reasoningBudget)
        ? Math.max(
            -1,
            Math.min(2_147_483_647, Math.trunc(partial.reasoningBudget)),
          )
        : DEFAULT_PER_MODEL_CONFIG.reasoningBudget,
    reasoningBudgetMessage:
      typeof partial.reasoningBudgetMessage === "string" &&
      isReasoningBudgetMessageValid(partial.reasoningBudgetMessage)
        ? partial.reasoningBudgetMessage
        : DEFAULT_PER_MODEL_CONFIG.reasoningBudgetMessage,
    tensorParallel:
      typeof partial.tensorParallel === "boolean"
        ? partial.tensorParallel
        : DEFAULT_PER_MODEL_CONFIG.tensorParallel,
    disableVision:
      typeof partial.disableVision === "boolean"
        ? partial.disableVision
        : DEFAULT_PER_MODEL_CONFIG.disableVision,
    chatTemplateOverride:
      typeof partial.chatTemplateOverride === "string" &&
      isChatTemplateWithinLimit(partial.chatTemplateOverride)
        ? partial.chatTemplateOverride
        : null,
    llamaExtraArgs: normalizeLlamaExtraArgs(partial.llamaExtraArgs),
    ...normalizeGpuFields(partial),
  };
}

/** The UI carries sentinels storage does not ("auto" -> null), which would read as non-default. */
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

/** Oldest version that understands every present field, so older clients can still rewrite it.
  Only a TRUE disableVision needs v4; same for the tuning group and reasoning pair. */
function storedSchemaVersion(normalized: PerModelConfig): number {
  if (normalized.mlxInt8Prefill) {
    return STORAGE_SCHEMA_VERSION;
  }
  if (normalized.mlxKvQuant != null) {
    return PRE_MLX_INT8_PREFILL_SCHEMA_VERSION;
  }
  const hasReasoningBudget =
    normalized.reasoningBudget !== -1 || normalized.reasoningBudgetMessage !== "";
  if (hasReasoningBudget) {
    return PRE_MLX_KV_QUANT_SCHEMA_VERSION;
  }
  const hasServerTuning =
    normalized.loadMode != null ||
    normalized.specDraftCacheDtype != null ||
    normalized.ctxCheckpoints != null ||
    normalized.cacheRam != null;
  if (hasServerTuning) {
    return PRE_REASONING_BUDGET_SCHEMA_VERSION;
  }
  if (normalized.disableVision) {
    return PRE_SERVER_TUNING_SCHEMA_VERSION;
  }
  if (normalized.llamaExtraArgs != null && normalized.llamaExtraArgs.length > 0) {
    return PRE_VISION_SCHEMA_VERSION;
  }
  if (normalized.nBatch != null || normalized.nUbatch != null) {
    return PRE_EXTRA_ARGS_SCHEMA_VERSION;
  }
  return PRE_BATCH_SCHEMA_VERSION;
}

function toStoredConfig(config: PerModelConfig): StoredPerModelConfig {
  const normalized = normalize(config);
  return {
    version: storedSchemaVersion(normalized),
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
  if (storedConfigVersion(map[key]) > STORAGE_SCHEMA_VERSION) {
    return null;
  }
  return normalize(map[key]);
}

export function resolveOnlyRememberedGgufVariant(
  modelId: string,
): { ggufVariant: string; config: PerModelConfig } | null {
  const map = readMap();
  const variants = new Map<string, string>();
  const normalizedModelId = normalizeModelIdentity(modelId);
  for (const key of Object.keys(map)) {
    const storedModelId = modelIdFromStorageKey(key);
    const ggufVariant = ggufVariantFromStorageKey(key);
    if (
      !storedModelId ||
      !ggufVariant ||
      normalizeModelIdentity(storedModelId) !== normalizedModelId
    ) {
      continue;
    }
    const normalizedVariant = normalizeGgufVariantIdentity(ggufVariant);
    if (normalizedVariant) {
      variants.set(normalizedVariant, ggufVariant);
    }
  }
  if (variants.size !== 1) {
    return null;
  }
  const ggufVariant = variants.values().next().value;
  if (!ggufVariant) {
    return null;
  }
  const key = findConfigKeyForModelVariant(map, modelId, ggufVariant);
  if (!key || storedConfigVersion(map[key]) > STORAGE_SCHEMA_VERSION) {
    return null;
  }
  return { ggufVariant, config: normalize(map[key]) };
}

export function isDefaultConfig(config: PerModelConfig): boolean {
  return (
    (config.engine ?? "auto") === "auto" &&
    (config.enginePrecision ?? "auto") === "auto" &&
    (config.engineParallelism ?? "tensor") === "tensor" &&
    config.customContextLength == null &&
    config.maxSeqLength == null &&
    (config.kvCacheDtype ?? null) === DEFAULT_PER_MODEL_CONFIG.kvCacheDtype &&
    (config.mlxKvQuant ?? null) === DEFAULT_PER_MODEL_CONFIG.mlxKvQuant &&
    !config.mlxInt8Prefill &&
    config.speculativeType === DEFAULT_PER_MODEL_CONFIG.speculativeType &&
    config.specDraftNMax == null &&
    config.nParallel == null &&
    config.reasoningBudget === DEFAULT_PER_MODEL_CONFIG.reasoningBudget &&
    config.reasoningBudgetMessage ===
      DEFAULT_PER_MODEL_CONFIG.reasoningBudgetMessage &&
    config.nBatch == null &&
    config.nUbatch == null &&
    // Compared against null: 0 checkpoints and 0 or -1 cache are values, and a default-judged entry is deleted.
    (config.specDraftCacheDtype ?? null) === null &&
    (config.loadMode ?? null) === null &&
    config.ctxCheckpoints == null &&
    config.cacheRam == null &&
    Boolean(config.tensorParallel) ===
      Boolean(DEFAULT_PER_MODEL_CONFIG.tensorParallel) &&
    Boolean(config.disableVision) ===
      Boolean(DEFAULT_PER_MODEL_CONFIG.disableVision) &&
    (config.chatTemplateOverride ?? null) === null &&
    // Or an Extra Arguments-only change reads as default and is deleted.
    (config.llamaExtraArgs == null || config.llamaExtraArgs.length === 0) &&
    gpuFieldsAtDefault(config)
  );
}

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
  /** Eviction is silent, so report evicted models or their server overrides cannot be forgotten. */
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
  if (hasFutureConfigForModelVariant(map, modelId, ggufVariant)) {
    return false;
  }
  if (!deleteConfigEntriesForModelVariant(map, modelId, ggufVariant)) {
    return true;
  }
  return writeMap(map);
}

/** Matched, not split: a colon is legal in a path, and two records can spell one key. */
function findModelOverrideKeyOwners(
  overrideKey: string,
): { modelId: string; ggufVariant: string | null }[] {
  const key = overrideKey.trim();
  const foldedKey = normalizeModelIdentity(key);
  const owners: { modelId: string; ggufVariant: string | null }[] = [];
  for (const storageKey of Object.keys(readMap())) {
    const modelId = modelIdFromStorageKey(storageKey);
    if (!modelId) {
      continue;
    }
    // Already lowercased by modelStorageKey, so only the tail folds below.
    const variant = normalizeGgufVariantIdentity(
      ggufVariantFromStorageKey(storageKey),
    );
    if (!variant) {
      if (foldedKey === normalizeModelIdentity(modelId)) {
        owners.push({ modelId, ggufVariant: null });
      }
      continue;
    }
    const cut = key.length - variant.length - 1;
    if (
      cut > 0 &&
      key[cut] === ":" &&
      key.slice(cut + 1).toLowerCase() === variant &&
      normalizeModelIdentity(key.slice(0, cut)) ===
        normalizeModelIdentity(modelId)
    ) {
      owners.push({ modelId, ggufVariant: variant });
    }
  }
  return owners;
}

export function deletePerModelConfigsForOverrideKeys(
  overrideKeys: readonly string[],
): boolean {
  let deleted = true;
  for (const overrideKey of overrideKeys) {
    for (const owner of findModelOverrideKeyOwners(overrideKey)) {
      if (!deletePerModelConfig(owner.modelId, owner.ggufVariant)) {
        deleted = false;
      }
    }
  }
  return deleted;
}

/**
 * Rename a config keyed by an older release's snapshot path to the repo id, in one write: saving
 * then deleting could exceed the budget and silently evict an unrelated model.
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
  // If the rename tips the map past its byte cap, leave storage as is: eviction is not undoable.
  const bytesAfter = serializedMapSize(map);
  if (
    bytesAfter > bytesBefore &&
    bytesAfter > MAX_PER_MODEL_CONFIG_STORAGE_BYTES
  ) {
    return false;
  }
  return writeMap(map);
}

export interface ResolvedPerModelConfig {
  config: PerModelConfig;
  remembered: boolean;
}

export function perModelConfigStorageChanged(
  atStart: ResolvedPerModelConfig,
  current: ResolvedPerModelConfig,
): boolean {
  return (
    atStart.remembered !== current.remembered ||
    JSON.stringify(toStoredConfig(atStart.config)) !==
      JSON.stringify(toStoredConfig(current.config))
  );
}

export function resolveInitialConfig(
  modelId: string,
  ggufVariant?: string | null,
): ResolvedPerModelConfig {
  const saved = loadPerModelConfig(modelId, ggufVariant);
  if (saved) {
    return { config: saved, remembered: true };
  }
  return { config: { ...DEFAULT_PER_MODEL_CONFIG }, remembered: false };
}

export function adoptCachedRepoConfig(
  modelId: string,
  ggufVariant?: string | null,
): string | null {
  const repoId = cachedRepoConfigId(modelId, ggufVariant);
  if (repoId) {
    adoptLegacyConfigKey(repoId, modelId, null);
  }
  return repoId;
}

/** An API auto-switch loads by snapshot path while settings key on repo id; only a namespaced
  collapse is adopted, per residentModelIdMatches. */
export function resolveResidentInitialConfig(
  modelId: string,
  ggufVariant?: string | null,
): ResolvedPerModelConfig {
  const repoId = adoptCachedRepoConfig(modelId, ggufVariant);
  if (repoId) {
    return resolveInitialConfig(repoId, null);
  }
  // A standalone file's reported quant is a label; its settings are saved without a variant.
  const standalone = isStandaloneGgufPath(modelId);
  const direct = resolveInitialConfig(modelId, standalone ? null : ggufVariant);
  if (direct.remembered) {
    return direct;
  }
  // Older pickers keyed loose .gguf settings by the label, and those records still exist.
  if (standalone && ggufVariant) {
    const labelled = resolveInitialConfig(modelId, ggufVariant);
    if (labelled.remembered) {
      return labelled;
    }
  }
  const alias = publicModelId(modelId);
  if (alias === modelId || !alias.includes("/")) {
    return direct;
  }
  return resolveInitialConfig(alias, ggufVariant);
}
