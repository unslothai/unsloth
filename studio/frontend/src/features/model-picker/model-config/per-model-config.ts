// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GpuIndexKind } from "@/hooks/use-gpu-info";
import {
  ggufVariantFromStorageKey,
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
  customContextLength: number | null;
  maxSeqLength: number | null;
  kvCacheDtype: string | null;
  /** MLX KV cache quantization width. Optional so older blobs still parse. */
  mlxKvBits?: number | null;
  speculativeType: string | null;
  specDraftNMax: number | null;
  /** KV cache dtype for the DRAFT context (--spec-draft-type-k/-v), sized and
   *  quantized independently of kvCacheDtype. Optional so older blobs parse. */
  specDraftCacheDtype?: string | null;
  nParallel: number | null;
  nBatch: number | null;
  nUbatch: number | null;
  /** --load-mode; null lets the fit decide: `none` when the load fits in VRAM
   *  (or VRAM plus host RAM), else no flag. Any value set here wins. */
  loadMode?: string | null;
  /** --ctx-checkpoints; null follows the llama.cpp default (32). */
  ctxCheckpoints?: number | null;
  /** --cache-ram in MiB; null follows the llama.cpp default (8192). */
  cacheRam?: number | null;
  tensorParallel: boolean;
  /** Load a vision GGUF without its mmproj, freeing the projector's VRAM. */
  disableVision: boolean;
  chatTemplateOverride: string | null;
  /**
   * Pass-through llama-server args, one argv token per entry, appended after
   * Unsloth's own flags.
   *
   * Three states, and the difference is load-bearing. `undefined` means this copy
   * never read the stored value, so a save must leave the server's alone: the
   * overrides route preserves the field when it is omitted, which is what kept
   * CLI-set flags alive while the panel had no control for them. `null` means the
   * user cleared the box, which has to be sent as an explicit `[]` or the clear is
   * silently dropped. A non-empty list is what to launch with.
   */
  llamaExtraArgs?: string[] | null;
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
  specDraftCacheDtype: null,
  nParallel: null,
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
// App default for a model no local backend sizes itself. Every path falls back here, not
// to an active model's runtime value, so an unconfigured pane never inherits and OOMs.
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

/** Whether MLX serves the RESIDENT model.
 *
 *  The platform cannot answer it alone: the worker picks NativeAudioBackend for a
 *  native-audio checkpoint before the MLX fast-path, so those load on Apple Silicon
 *  without MLX serving them. `loadedIsMlx` is the backend's own answer, and null there
 *  means nothing is loaded yet, where the platform is still the best available one.
 */
export function residentIsServedByMlx(
  isGguf: boolean,
  deviceType: string | null | undefined,
  chatOnlyReason: string | null | undefined,
  loadedIsMlx: boolean | null | undefined,
): boolean {
  return isServedByMlx(isGguf, deviceType, chatOnlyReason) && loadedIsMlx !== false;
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

/** Whether llama.cpp serves the active model.
 *
 *  `loadedIsGguf` is the backend's own answer; the rest identify a GGUF that has not
 *  reported one yet. A context length is not among them -- MLX reports one too, so it
 *  says a model is loaded, not which backend loaded it. An external provider is excluded
 *  because its id keeps a `.gguf` suffix while the flag describes a local load.
 */
export function isServedByLlamaCpp(x: {
  loadedIsGguf?: boolean | null;
  activeGgufVariant?: string | null;
  activeNativePathToken?: string | null;
  checkpoint?: string | null;
}): boolean {
  if (isExternalModelId(x.checkpoint)) return false;
  // A reported non-GGUF backend settles it: the variant and path token outlive the pick
  // that set them, so neither is evidence past a reload.
  if (x.loadedIsGguf === false) return false;
  return (
    x.loadedIsGguf === true ||
    x.activeGgufVariant != null ||
    x.activeNativePathToken != null ||
    String(x.checkpoint ?? "").toLowerCase().endsWith(".gguf")
  );
}

/** The store's record of the context window a load left behind.
 *
 *  A window counts when the backend that reported it sized one. MLX always does, so its
 *  `context_length` stands alone even with no trained window in the config. Transformers
 *  sizes nothing and echoes the requested max_seq_length, so without a native length it
 *  contributes no window.
 *
 *  One constructor because the four move together: a window without the backend that
 *  produced it is what made a context length read as proof of a GGUF.
 */
export function loadedContextFields(resp: {
  is_gguf?: boolean;
  is_mlx?: boolean;
  context_length?: number | null;
  native_context_length?: number | null;
  max_context_length?: number | null;
  context_length_enforced?: boolean | null;
} | null): {
  loadedContextLength: number | null;
  maxContextLength: number | null;
  nativeContextLength: number | null;
  loadedIsGguf: boolean | null;
  loadedIsMlx: boolean | null;
  loadedContextEnforced: boolean | null;
} {
  if (!resp) {
    return {
      loadedContextLength: null,
      maxContextLength: null,
      nativeContextLength: null,
      loadedIsGguf: null,
      loadedIsMlx: null,
      loadedContextEnforced: null,
    };
  }
  const isGguf = resp.is_gguf ?? false;
  // Unknown, not a default: a response omits it when reading the model's window failed.
  const loaded = resp.context_length ?? null;
  if (!isGguf && !resp.is_mlx && resp.native_context_length == null) {
    return {
      loadedContextLength: null,
      maxContextLength: null,
      nativeContextLength: null,
      loadedIsGguf: false,
      loadedIsMlx: resp.is_mlx ?? null,
      loadedContextEnforced: null,
    };
  }
  return {
    loadedContextLength: loaded,
    maxContextLength: resp.max_context_length ?? loaded,
    nativeContextLength: resp.native_context_length ?? null,
    loadedIsGguf: isGguf,
    // The backend's own answer, so a checkpoint the worker serves off the MLX path
    // (native audio) is not taken for MLX by the platform alone.
    loadedIsMlx: resp.is_mlx ?? null,
    // llama.cpp allocates what it reports, so GGUF is enforced by construction.
    // Everything else answers for itself, or says nothing.
    loadedContextEnforced: isGguf ? true : (resp.context_length_enforced ?? null),
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

// Every width mx.quantize supports. By bit width, not a dtype name, hence separate
// from KV_CACHE_DTYPES.
export const MLX_KV_BITS: readonly number[] = [8, 6, 5, 4, 3, 2];
const VALID_KV_CACHE_DTYPES = new Set<string>(KV_CACHE_DTYPES);

// llama-server's --load-mode enum, in --help order. "auto" is the default: the UI
// shows it, storage keeps null and the backend emits no flag, so the fit may pick
// "none". Never sent verbatim; builds like b10360 reject "auto" as a value.
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

// --ctx-checkpoints: per-slot snapshots of the sliding-window cache, so the count
// is small. 0 disables them; the ceiling is a sanity bound, not an upstream one.
export const CTX_CHECKPOINTS_MIN = 0;
export const CTX_CHECKPOINTS_MAX = 256;
export const CTX_CHECKPOINTS_LLAMA_DEFAULT = 32;

// --cache-ram in MiB: -1 is "no limit" and 0 disables the host prompt cache, so
// the floor is -1. The 1 TiB ceiling fails a stray keystroke before the child does.
export const CACHE_RAM_MIN = -1;
export const CACHE_RAM_MAX = 1024 * 1024;
export const CACHE_RAM_LLAMA_DEFAULT = 8192;

export {
  DRAFT_N_MAX_SPEC_TYPES,
  SEPARATE_DRAFT_MODEL_SPEC_TYPES,
  SPECULATIVE_TYPES,
} from "@/lib/speculative-modes";

/** Exported so cross-tab listeners can tell this key's storage event from the
 *  dozens of others Studio writes. */
export const PER_MODEL_CONFIG_STORAGE_KEY = "unsloth_model_configs";
const STORAGE_KEY = PER_MODEL_CONFIG_STORAGE_KEY;
const LEGACY_STORAGE_KEY = "unsloth_load_settings";
const LEGACY_MIGRATION_FLAG = "unsloth_model_configs_migrated";
// v2 added nBatch / nUbatch, v3 llamaExtraArgs, v4 disableVision and v5 the
// llama-server tuning group (loadMode / specDraftCacheDtype / ctxCheckpoints /
// cacheRam); a client from before any of them would normalize the field it does
// not know straight back out of the record.
const STORAGE_SCHEMA_VERSION = 5;
const PRE_SERVER_TUNING_SCHEMA_VERSION = 4;
const PRE_VISION_SCHEMA_VERSION = 3;
const PRE_EXTRA_ARGS_SCHEMA_VERSION = 2;
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
  "specDraftCacheDtype",
  "nParallel",
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

/**
 * Keep only a list of strings, preserving the three states above.
 *
 * Anything that is not an array is "not loaded" (`undefined`), never "cleared":
 * the wiping case is the one worth being careful about, since the flags it would
 * throw away are invisible in this panel until the row reads them.
 */
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

/**
 * Canonicalize a stored --load-mode, or null to follow the llama.cpp default.
 *
 * "auto" folds to null like Speculative Decoding's: it IS the default, so storing
 * it would pin a value the build may redefine and read as an override everywhere.
 */
export function canonicalizeLoadMode(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  // Whitespace and case only: "mmap + mlock" is not a spelling llama-server
  // accepts, so it is refused rather than repaired.
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

/** The context a saved record pins, whichever field it was written in.
 *
 *  MLX's pin moved into `customContextLength` beside llama.cpp's, while transformers
 *  keeps its own in `maxSeqLength` and a record written before the move still carries an
 *  MLX pin there. The same record is read on a host that serves it with a different
 *  backend, so every read has to honour both.
 *
 *  A *saved* record only. `currentRuntimePerModelConfig` builds the same shape from the
 *  live store, where `maxSeqLength` is the length the model resolved to rather than one
 *  anybody chose; read that through `customContextLength` alone.
 */
export function savedContextPin(config: {
  customContextLength?: number | null;
  maxSeqLength?: number | null;
}): number | null {
  return config.customContextLength ?? normalizeMaxSeqLength(config.maxSeqLength ?? null);
}

/** The patch that pins a context for a non-GGUF target, on the backend serving it.
 *
 *  An edit leaves a pin in exactly one field, clearing whichever the record arrived with:
 *  a record holding both loads at a different length depending on who asked -- the picker
 *  resolves `customContextLength` first, the API's auto-switch load `maxSeqLength`.
 */
export function contextPinPatch(value: number, isMlx: boolean): Partial<PerModelConfig> {
  // Held to what a load may ask for, but not rounded to the control's step: a pin taken
  // from a resolved window need not sit on that grid.
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
    disableVision:
      typeof raw.disableVision === "boolean" ? raw.disableVision : false,
    chatTemplateOverride: null,
    // Absent, not null: a legacy blob predates the editor, and the server may well
    // hold flags set from the CLI. Reading that as "cleared" would wipe them on the
    // first save from this panel.
    llamaExtraArgs: undefined,
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

/** Fires when any model's saved config changes, in this tab. The browser's own
 *  `storage` event only reaches *other* tabs, so readers that need to react to
 *  an edit made here (the picker's memory bar) have nothing else to listen to. */
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
  // Best-effort: the write already landed, so a host that cannot dispatch
  // events must not make a saved config report back as unsaved.
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
  // Tied to the mode like specDraftNMax: a dtype stored under a mode with no
  // separate drafter shows a row for a context that never exists.
  const specDraftCacheDtype =
    speculativeType != null &&
    SEPARATE_DRAFT_MODEL_SPEC_TYPES.has(speculativeType) &&
    typeof partial.specDraftCacheDtype === "string" &&
    VALID_KV_CACHE_DTYPES.has(partial.specDraftCacheDtype)
      ? partial.specDraftCacheDtype
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
  // Stamped with the OLDEST version that still understands every field present, so
  // a record an older client can safely rewrite is not needlessly locked away from
  // it. Only a record carrying a newer field is put out of that client's reach.
  // Only a TRUE disableVision needs the v4 stamp. The default is false, which
  // is what a pre-vision client reconstructs anyway, so a record that merely
  // carries the key at its default loses nothing by staying in that client's
  // reach -- and stamping every record v4 would put the whole store out of it.
  // The tuning group above it follows the same rule: only a record that actually
  // sets one of the four is put out of a pre-v5 client's reach.
  const hasServerTuning =
    normalized.loadMode != null ||
    normalized.specDraftCacheDtype != null ||
    normalized.ctxCheckpoints != null ||
    normalized.cacheRam != null;
  const version = hasServerTuning
    ? STORAGE_SCHEMA_VERSION
    : normalized.disableVision
      ? PRE_SERVER_TUNING_SCHEMA_VERSION
      : normalized.llamaExtraArgs != null && normalized.llamaExtraArgs.length > 0
        ? PRE_VISION_SCHEMA_VERSION
        : normalized.nBatch != null || normalized.nUbatch != null
          ? PRE_EXTRA_ARGS_SCHEMA_VERSION
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
    config.customContextLength == null &&
    config.maxSeqLength == null &&
    (config.kvCacheDtype ?? null) === DEFAULT_PER_MODEL_CONFIG.kvCacheDtype &&
    (config.mlxKvBits ?? null) === DEFAULT_PER_MODEL_CONFIG.mlxKvBits &&
    config.speculativeType === DEFAULT_PER_MODEL_CONFIG.speculativeType &&
    config.specDraftNMax == null &&
    config.nParallel == null &&
    config.nBatch == null &&
    config.nUbatch == null &&
    // The llama-server tuning group, for the same reason as the arguments below:
    // savePerModelConfig deletes an entry it judges default, so a config whose only
    // change was one of these was dropped on the way to storage while the settings
    // page reported that defaults were kept. Compared against null, not truth: 0
    // checkpoints and a 0 or -1 cache are values, not blanks.
    (config.specDraftCacheDtype ?? null) === null &&
    (config.loadMode ?? null) === null &&
    config.ctxCheckpoints == null &&
    config.cacheRam == null &&
    Boolean(config.tensorParallel) ===
      Boolean(DEFAULT_PER_MODEL_CONFIG.tensorParallel) &&
    Boolean(config.disableVision) ===
      Boolean(DEFAULT_PER_MODEL_CONFIG.disableVision) &&
    (config.chatTemplateOverride ?? null) === null &&
    // Or a config whose only change is Extra Arguments reads as default, and
    // savePerModelConfig deletes the entry it was asked to remember.
    (config.llamaExtraArgs == null || config.llamaExtraArgs.length === 0) &&
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
): ResolvedPerModelConfig {
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
