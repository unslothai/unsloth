// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Server-side mirror of the per-model config. ../model-config/per-model-config.ts lives in
// browser localStorage, so an API auto-switch load (no browser in the loop) came up with none of
// the user's settings. routes/inference.py reads this map and rebuilds the picker's LoadRequest.

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
  splitQuantSuffix,
} from "../model-config/model-identity";
import {
  DEFAULT_PER_MODEL_CONFIG,
  type PerModelConfig,
  normalizePerModelConfig,
} from "../model-config/per-model-config";

const OVERRIDES_URL = "/api/settings/openai-auto-switch/overrides";

/** One model's stored launch config, as the backend persists it. */
export interface ApiModelOverride {
  // biome-ignore lint/style/useNamingConvention: API schema
  llama_extra_args?: string[];
  // biome-ignore lint/style/useNamingConvention: API schema
  max_seq_length?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  custom_context_length?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  kv_cache_dtype?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  mlx_kv_bits?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  speculative_type?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  spec_draft_n_max?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  n_parallel?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  n_batch?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  n_ubatch?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  load_mode?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  spec_draft_cache_type?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  ctx_checkpoints?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  cache_ram?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  tensor_parallel?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  disable_vision?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  chat_template_override?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  gpu_memory_mode?: "auto" | "manual";
  // biome-ignore lint/style/useNamingConvention: API schema
  gpu_layers?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  n_cpu_moe?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  gpu_ids?: number[];
}

export type ApiModelOverrides = Record<string, ApiModelOverride>;

/**
 * The key one model's config is stored under: the `repo:VARIANT` form an OpenAI
 * request names a quant by, so two quants of one repo keep separate configs and the
 * backend matches the requested name directly. Bare id when there is no variant.
 */
export function modelOverrideKey(
  modelId: string,
  ggufVariant?: string | null,
): string {
  return ggufVariant ? `${modelId}:${ggufVariant}` : modelId;
}

/**
 * A path that names one file whatever the casing, folded for comparison, or null
 * when the path is case-sensitive.
 *
 * The rules are _fold_case_insensitive_path's, and they have to be, because the
 * server applies an override this resolver has to find: a Windows drive path, a UNC
 * share and a WSL drive mount all fold, the separator is interchangeable, and
 * trailing separators are trimmed down to the root. A POSIX path is not folded, or
 * "/models/Foo.gguf" would collect the arguments stored for "/models/foo.gguf".
 */
const WINDOWS_DRIVE_PATH = /^[a-zA-Z]:[\\/]/;
const WSL_DRIVE_PATH = /^\/mnt\/[a-zA-Z](\/|$)/;

function foldCaseInsensitivePath(key: string): string | null {
  const slashed = key.replace(/\\/g, "/");
  let minimum: number;
  if (WINDOWS_DRIVE_PATH.test(key)) {
    minimum = 3;
  } else if (slashed.startsWith("//")) {
    minimum = 2;
  } else if (WSL_DRIVE_PATH.test(slashed)) {
    minimum = 6;
  } else {
    return null;
  }
  let trimmed = slashed;
  while (trimmed.length > minimum && trimmed.endsWith("/")) {
    trimmed = trimmed.slice(0, -1);
  }
  return trimmed.toLowerCase();
}

function foldOverrideKey(key: string): string {
  // A path that folds does so whole, separators and casing together.
  const path = foldCaseInsensitivePath(key);
  if (path !== null) {
    return path;
  }
  // splitQuantSuffix, not the last colon: a colon is legal in a POSIX filename, so
  // "/models/foo:Bar.gguf" is a whole path and reading "Bar.gguf" as a quant would
  // fold it onto the real, different file "/models/foo:bar.gguf". This is the check
  // the backend's split_quant_suffix makes.
  const split = splitQuantSuffix(key);
  const id = split ? split[0] : key;
  const quant = split ? `:${split[1].toLowerCase()}` : "";
  // A POSIX path keeps its case; only the quant folds, because the browser
  // lowercases that before storing. A repo id folds whole.
  return id.startsWith("/") ? `${id}${quant}` : `${id.toLowerCase()}${quant}`;
}

/**
 * The stored arguments, with "the row carried an empty list" kept apart from "no
 * row carried the field at all".
 *
 * The distinction is what the settings page's tombstone is for: clearing the box
 * for one quant saves an explicit [], and that is what stops the server's lookup
 * before it reaches a legacy bare-repository row that still holds arguments.
 * Collapsing the two left the panel with llamaExtraArgs undefined, its next Load
 * omitted the field, and /load carried the resident model's arguments over: the
 * very ones that had just been cleared.
 */
export type ResolvedExtraArgs = {
  tokens: string[];
  explicit: boolean;
};

/** The field as one entry stores it, empty list and absent field kept apart. */
function resolvedFrom(entry: ApiModelOverride): ResolvedExtraArgs {
  const tokens = entry.llama_extra_args;
  return { tokens: tokens ?? [], explicit: Array.isArray(tokens) };
}

function presentOverride(
  value: ApiModelOverride | undefined | null,
): ApiModelOverride | null {
  return value && Object.keys(value).length > 0 ? value : null;
}

export function resolveStoredOverride(
  overrides: ApiModelOverrides,
  keys: readonly string[],
): ApiModelOverride | null {
  // The overrides route folds identities before it reads a row: a repo id and a
  // quant differing only in case resolve to the same entry, and it falls back from
  // `repo:QUANT` to the bare repo. Keys are tried most specific first, then folded.
  // Whole ENTRIES, in the order the backend tries them, stopping at the first one
  // that exists. The auto-switch loader breaks on the first non-empty override and
  // reads its fields from there, so falling through to a bare repo id because the
  // variant row happens to carry no arguments would launch the picker with flags an
  // API load would not use.
  // An entry with no fields is skipped rather than stopping the search, because
  // that is what `if override: break` does on the server.
  const folded = new Map<string, ApiModelOverride | null>();
  for (const [key, value] of Object.entries(overrides)) {
    if (!presentOverride(value)) {
      continue;
    }
    const foldedKey = foldOverrideKey(key);
    // Ambiguous folds resolve to nothing, as resolve_model_override_key does:
    // duplicate case variants left by an upgrade must not have one of them picked
    // at enumeration order, which is another model's settings half the time.
    folded.set(foldedKey, folded.has(foldedKey) ? null : value);
  }
  for (const key of keys) {
    const exact = presentOverride(overrides[key]);
    if (exact) {
      return exact;
    }
    // Folding, by the same rule the backend resolves with: a POSIX path is
    // case-sensitive, so lowercasing one whole would hand /models/foo.gguf the
    // arguments stored for /models/Foo.gguf and then send them explicitly on Load.
    // Windows, UNC and WSL paths do fold, and so does the quant suffix the browser
    // lowercases before storing.
    const match = folded.get(foldOverrideKey(key));
    if (match) {
      return match;
    }
  }
  return null;
}

export function resolveStoredExtraArgs(
  overrides: ApiModelOverrides,
  keys: readonly string[],
): ResolvedExtraArgs {
  const resolved = resolveStoredOverride(overrides, keys);
  return resolved ? resolvedFrom(resolved) : { tokens: [], explicit: false };
}

export async function fetchModelOverrides(): Promise<ApiModelOverrides> {
  const res = await authFetch(OVERRIDES_URL);
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load saved model settings"),
    );
  }
  const body = (await res.json()) as { overrides?: ApiModelOverrides };
  return body.overrides ?? {};
}

/**
 * The pass-through arguments the LOAD of this model would apply.
 *
 * Asked of the server rather than worked out here, because the resolution is
 * Python's: casefold is not toLowerCase (Straße and STRASSE are one path to the
 * loader and two to a browser), and an ambiguous fold deliberately matches nothing.
 * A client mirroring those rules can only approximate them, and the approximation
 * shows up as a cold load launching without arguments an API load applies.
 *
 * Falls back to resolving locally against the whole map when the backend predates
 * the parameter, which is the same answer in every case but the exotic ones.
 */
export async function fetchLoadModelOverride(
  loadId: string,
  aliasId: string,
  ggufVariant?: string | null,
  fallbackKeys: readonly string[] = [],
): Promise<ApiModelOverride | null> {
  const query = new URLSearchParams({ model_id: loadId, alias_id: aliasId });
  if (ggufVariant) {
    query.set("gguf_variant", ggufVariant);
  }
  const res = await authFetch(`${OVERRIDES_URL}?${query.toString()}`);
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load saved model settings"),
    );
  }
  const body = (await res.json()) as {
    overrides?: ApiModelOverrides;
    resolved?: ApiModelOverride | null;
    // biome-ignore lint/style/useNamingConvention: API schema
    resolved_key?: string | null;
  };
  if (body.resolved !== undefined) {
    return presentOverride(body.resolved);
  }
  // A backend that predates the resolved field answers with the whole map, and the
  // caller has to say which keys to look under. Derived here when it did not: the
  // panel passes its own richer list (it knows the alias the settings page used),
  // while the auto-load and compare callers have only these two identities, and
  // defaulting to none made them read an empty map and launch without the stored
  // arguments against an older server.
  const derived =
    fallbackKeys.length > 0
      ? fallbackKeys
      : [
          modelOverrideKey(loadId, ggufVariant),
          modelOverrideKey(aliasId, ggufVariant),
          loadId,
          aliasId,
        ].filter((key, index, all) => all.indexOf(key) === index);
  return resolveStoredOverride(body.overrides ?? {}, derived);
}

export async function fetchLoadExtraArgs(
  loadId: string,
  aliasId: string,
  ggufVariant?: string | null,
  fallbackKeys: readonly string[] = [],
): Promise<ResolvedExtraArgs> {
  const resolved = await fetchLoadModelOverride(
    loadId,
    aliasId,
    ggufVariant,
    fallbackKeys,
  );
  // An explicit empty list is a cleared box, not an absence, and the caller has
  // to send it as one: omitting the field lets /load carry the resident model's
  // arguments over, which is exactly what was cleared.
  return resolvedFrom(resolved ?? {});
}

/**
 * Translate one server-resolved override into the config shape the picker uses.
 *
 * The row is authoritative for the fields it CARRIES, and only for those. An absent
 * field is not evidence the user chose the default: the mirror is best-effort and
 * lossy, so a PUT that never landed, a value this build's normalizer refused, and a
 * row written before a field reached the route all leave the same gap. `localConfig`
 * fills it, or opening the panel would delete settings the user typed here -- a
 * migrated legacy config, or one whose mirror failed, came back as bare defaults and
 * was then persisted over the original. The cost is that clearing ONE field on another
 * origin does not travel until this one saves again, which the schema cannot express
 * anyway: an omitted field and an app default are written the same way.
 */
export function fromApiOverride(
  override: ApiModelOverride,
  localConfig?: PerModelConfig,
): PerModelConfig {
  const local = localConfig ?? DEFAULT_PER_MODEL_CONFIG;
  // Three states, so the winning source is chosen once and restored below: the row's
  // list when it has one, else whatever this browser held.
  const extraArgs = Array.isArray(override.llama_extra_args)
    ? override.llama_extra_args
    : local.llamaExtraArgs;
  // Only a physical pin travels (toApiOverride drops the rest), so a row without ids
  // says nothing about placement and a local Vulkan ordinal keeps its own namespace.
  const serverGpuIds = override.gpu_ids?.length ? override.gpu_ids : null;
  // The pin is ONE setting in one of two fields, and an edit clears the other
  // (contextPinPatch). Filling them from different sources mints a record that loads at
  // two lengths, since the picker reads customContextLength first and the API's
  // auto-switch max_seq_length first. So a row stating either field owns both.
  const serverStatesPin =
    override.custom_context_length != null || override.max_seq_length != null;
  const normalized = normalizePerModelConfig({
    ...DEFAULT_PER_MODEL_CONFIG,
    customContextLength: serverStatesPin
      ? (override.custom_context_length ?? null)
      : local.customContextLength,
    maxSeqLength: serverStatesPin
      ? (override.max_seq_length ?? null)
      : local.maxSeqLength,
    kvCacheDtype: override.kv_cache_dtype ?? local.kvCacheDtype,
    mlxKvBits: override.mlx_kv_bits ?? local.mlxKvBits,
    speculativeType: override.speculative_type ?? local.speculativeType,
    specDraftNMax: override.spec_draft_n_max ?? local.specDraftNMax,
    specDraftCacheDtype:
      override.spec_draft_cache_type ?? local.specDraftCacheDtype,
    nParallel: override.n_parallel ?? local.nParallel,
    nBatch: override.n_batch ?? local.nBatch,
    nUbatch: override.n_ubatch ?? local.nUbatch,
    loadMode: override.load_mode ?? local.loadMode,
    ctxCheckpoints: override.ctx_checkpoints ?? local.ctxCheckpoints,
    cacheRam: override.cache_ram ?? local.cacheRam,
    // Both are stored only when true, so an absent one is a gap like any other.
    tensorParallel: override.tensor_parallel ?? local.tensorParallel,
    disableVision: override.disable_vision ?? local.disableVision,
    chatTemplateOverride:
      override.chat_template_override ?? local.chatTemplateOverride,
    llamaExtraArgs: extraArgs,
    gpuMemoryMode: override.gpu_memory_mode ?? local.gpuMemoryMode,
    gpuLayers: override.gpu_layers ?? local.gpuLayers,
    nCpuMoe: override.n_cpu_moe ?? local.nCpuMoe,
    selectedGpuIds: serverGpuIds ?? local.selectedGpuIds ?? null,
    selectedGpuIndexKind: serverGpuIds
      ? "physical"
      : (local.selectedGpuIndexKind ?? null),
  });
  // normalizePerModelConfig intentionally collapses an empty list to null. The server
  // uses [] as a tombstone that stops fallback to a broader override, so hydration
  // must retain that third state.
  if (Array.isArray(extraArgs)) {
    normalized.llamaExtraArgs = [...extraArgs];
  }
  return normalized;
}

/**
 * Translate the UI's per-model config into the backend's schema. Only fields the user set are
 * sent: an absent field reads as "app default", so nulls would pin defaults and stop the model
 * following later global changes. A `null` config clears the entry. Exported so the backfill
 * can compare it against the stored entry.
 */
export function toApiOverride(config: PerModelConfig | null): ApiModelOverride {
  if (!config) {
    return {};
  }
  const payload: ApiModelOverride = {};
  if (config.maxSeqLength && config.maxSeqLength > 0) {
    payload.max_seq_length = config.maxSeqLength;
  }
  if (config.customContextLength && config.customContextLength > 0) {
    payload.custom_context_length = config.customContextLength;
  }
  if (config.kvCacheDtype) {
    payload.kv_cache_dtype = config.kvCacheDtype;
  }
  // Travels beside kv_cache_dtype, or an API auto-switch loads a remembered MLX
  // model at full precision.
  if (config.mlxKvBits != null) {
    payload.mlx_kv_bits = config.mlxKvBits;
  }
  if (config.speculativeType) {
    payload.speculative_type = config.speculativeType;
  }
  if (config.specDraftNMax && config.specDraftNMax > 0) {
    payload.spec_draft_n_max = config.specDraftNMax;
  }
  // Blank follows the server-wide --parallel default, which is the app default here.
  if (config.nParallel && config.nParallel > 0) {
    payload.n_parallel = config.nParallel;
  }
  // blank follows the llama.cpp defaults (2048 / 512)
  if (config.nBatch && config.nBatch > 0) {
    payload.n_batch = config.nBatch;
  }
  if (config.nUbatch && config.nUbatch > 0) {
    payload.n_ubatch = config.nUbatch;
  }
  // Blank follows llama.cpp's own default for each: auto, f16, 32 and 8192.
  // cacheRam is compared against null rather than tested for truth, because 0 and
  // -1 are both meaningful values here (disable, and no limit).
  if (config.loadMode) {
    payload.load_mode = config.loadMode;
  }
  if (config.specDraftCacheDtype) {
    payload.spec_draft_cache_type = config.specDraftCacheDtype;
  }
  if (config.ctxCheckpoints != null && config.ctxCheckpoints >= 0) {
    payload.ctx_checkpoints = config.ctxCheckpoints;
  }
  if (config.cacheRam != null) {
    payload.cache_ram = config.cacheRam;
  }
  if (config.tensorParallel) {
    payload.tensor_parallel = true;
  }
  if (config.disableVision) {
    payload.disable_vision = true;
  }
  if (config.chatTemplateOverride?.trim()) {
    payload.chat_template_override = config.chatTemplateOverride;
  }
  // The one field where absent does NOT mean "app default": the route preserves
  // llama_extra_args it is not sent, which is what kept CLI-set flags alive while
  // this panel had no control for them. So `undefined` (never read) stays omitted,
  // and a cleared box has to say so with an explicit empty list.
  if (config.llamaExtraArgs !== undefined) {
    payload.llama_extra_args = config.llamaExtraArgs ?? [];
  }
  // Only "manual" is a real override; "auto" is the follow-the-global default.
  if (config.gpuMemoryMode === "manual") {
    payload.gpu_memory_mode = "manual";
  }
  // gpuLayers < 0 is Auto, which is also the default.
  if (typeof config.gpuLayers === "number" && config.gpuLayers >= 0) {
    payload.gpu_layers = config.gpuLayers;
  }
  if (typeof config.nCpuMoe === "number" && config.nCpuMoe > 0) {
    payload.n_cpu_moe = config.nCpuMoe;
  }
  // Only a physical pin travels. The same integers are a Vulkan ordinal under Vulkan and a
  // CUDA/ROCm index elsewhere, and the override carries no namespace, so after a backend change
  // the server would pin a different device with ids its availability check accepts. An absent
  // kind predates the field, which was physical-only. Dropping the pin leaves auto placement.
  const gpuIndexKind = config.selectedGpuIndexKind ?? "physical";
  if (
    config.selectedGpuIds &&
    config.selectedGpuIds.length > 0 &&
    gpuIndexKind === "physical"
  ) {
    payload.gpu_ids = config.selectedGpuIds;
  }
  return payload;
}

// One in-flight write per model, so writes commit in issue order: otherwise the older
// response can land last and resurrect the entry the newer one replaced. Models overlap.
const writesByKey = new Map<string, Promise<void>>();

export interface PutModelOverrideOptions {
  /**
   * Fill in only what is missing: every value already on the server stays as it is. The backfill
   * reads the map once then writes each model in turn, so another tab's save would be overwritten
   * by this browser's older copy. Field level, so a legacy entry gains the browser-only fields.
   */
  fillAbsentFields?: boolean;
  /**
   * Clear the fields this UI mirrors but leave the server's own launch flags alone. Evicting a
   * local entry for the storage budget is not a forget, so it must not take `llama_extra_args`
   * the page can neither show nor restore. The route drops the row once nothing is left in it.
   */
  keepLaunchFlags?: boolean;
}

export async function putModelOverride(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig | null,
  options?: PutModelOverrideOptions,
): Promise<void> {
  // Keyed by the folded identity: the backend resolves a legacy casing and the
  // normalized one to one row, so raw strings would open two queues and race again.
  const key = modelOverrideKey(
    normalizeModelIdentity(modelId),
    normalizeGgufVariantIdentity(ggufVariant),
  );
  // Chain on the settled tail: a failed write must not cancel the next one.
  const previous = writesByKey.get(key) ?? Promise.resolve();
  const write = previous
    .catch(() => {})
    .then(() => sendModelOverride(modelId, ggufVariant, config, options));
  writesByKey.set(key, write);
  try {
    await write;
  } finally {
    // Only the last writer clears the slot, so a queue still building keeps order.
    if (writesByKey.get(key) === write) {
      writesByKey.delete(key);
    }
  }
}

async function sendModelOverride(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig | null,
  options?: PutModelOverrideOptions,
): Promise<void> {
  const res = await authFetch(OVERRIDES_URL, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      model_id: modelOverrideKey(modelId, ggufVariant),
      // This build mirrors the llama-server tuning group, so an omission here is the
      // user clearing it rather than a client that predates the fields. Without this
      // the backend preserves the stored values, which is what stops a cached older
      // bundle from deleting settings it never knew to send. An older backend ignores
      // the key.
      // biome-ignore lint/style/useNamingConvention: API schema
      mirrors_server_tuning: true,
      // Only sent when set, so an older backend is not handed an unknown key every save.
      ...(options?.fillAbsentFields
        ? // biome-ignore lint/style/useNamingConvention: API schema
          { fill_absent_fields: true }
        : {}),
      // Say which operation this is: an all-default save carries no fields, shape-identical
      // to a forget, and guessing wrong wipes flags the UI cannot show or restore.
      remove: config === null && !options?.keepLaunchFlags,
      // The backend preserves the flags when omitted, which is what a save that
      // never opened the box relies on; a forget means all of it, so that path
      // sends an explicit [].
      ...(config === null && !options?.keepLaunchFlags
        ? // biome-ignore lint/style/useNamingConvention: API schema
          { llama_extra_args: [] }
        : {}),
      ...toApiOverride(config),
    }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to save model settings for the API"),
    );
  }
}

/**
 * Mirror a per-model config save to the backend without blocking the UI. Best-effort: the
 * localStorage write already happened, so a failed sync must not fail the save. Logged, not
 * toasted: an API load falls back to defaults until the next save.
 */
export function syncModelOverride(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig | null,
  options?: PutModelOverrideOptions,
): void {
  void putModelOverride(modelId, ggufVariant, config, options).catch(
    (error: unknown) => {
      console.warn(
        "Failed to mirror model settings to the server; an API load of this model will use defaults.",
        error,
      );
    },
  );
}
