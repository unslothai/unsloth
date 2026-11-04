// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// LLM cards edit their settings in Chat's panel. With "Use the model's saved settings" on they read
// Chat's per-model memory (paramsByModel, the same row the engine reads at run time in
// core/flows/model_settings.py); any edit forks the card onto its own copy. No React or store
// imports, so the rules stay unit-testable.

import type { ModelChoice, NodeSpec } from "../types";

export const LLM_SETTINGS_NODES: ReadonlySet<string> = new Set([
  "llm.generate",
  "llm.vision",
]);

export const USE_SAVED = "use_model_settings";

/** Rendered by the settings panel, not as generic inspector rows. */
export const PANEL_PARAMS: ReadonlySet<string> = new Set([
  USE_SAVED,
  "system",
  "temperature",
  "top_p",
  "top_k",
  "min_p",
  "repetition_penalty",
  "presence_penalty",
  "max_tokens",
  "seed",
  "thinking",
  "reasoning_effort",
  "json_schema",
  "load",
]);

/** Chat's saved launch settings for a model, in the override shape the server stores and loads with. A card keeps
 *  its own load settings in the same shape, so one mapping on the server turns either into the load request. */
export type LoadSettings = Record<string, unknown>;

/** The subset of Chat's remembered params a card maps onto. */
export interface SavedModelSettings {
  systemPrompt?: string;
  temperature?: number;
  topP?: number;
  topK?: number;
  minP?: number;
  minPMode?: "server-default" | "custom";
  repetitionPenalty?: number;
  presencePenalty?: number;
  maxTokens?: number;
}

type Params = Record<string, unknown>;

const SAVED_TO_PARAM = [
  ["systemPrompt", "system"],
  ["temperature", "temperature"],
  ["topP", "top_p"],
  ["topK", "top_k"],
  ["minP", "min_p"],
  ["repetitionPenalty", "repetition_penalty"],
  ["presencePenalty", "presence_penalty"],
  ["maxTokens", "max_tokens"],
] as const;

/** Chat's "Max" output cap: the load's whole window. A card takes its own ceiling, as the engine does. */
const MAX_TOKENS_AS_MAX = 2 ** 31 - 1;

const panelSpecs = new WeakMap<NodeSpec, NodeSpec>();

/** The spec without the panel's params, cached so a store selector stays stable. */
export function withoutPanelParams<T extends NodeSpec | undefined>(spec: T): T {
  if (!spec || !LLM_SETTINGS_NODES.has(spec.type)) return spec;
  let found = panelSpecs.get(spec);
  if (!found) {
    found = {
      ...spec,
      params: spec.params.filter((p) => !PANEL_PARAMS.has(p.id)),
    };
    panelSpecs.set(spec, found);
  }
  return found as T;
}

export function isOn(params: Params): boolean {
  return params[USE_SAVED] === true;
}

/** New cards start on: they have no values of their own to protect yet. Flows saved before this
 *  setting existed have no key and stay off, so they run with the values they were saved with. */
export function newCardParams(type: string, params: Params): Params {
  return LLM_SETTINGS_NODES.has(type) ? { ...params, [USE_SAVED]: true } : params;
}

/** The model a card with no model picked runs on: what the server reports resident (as the run resolves it),
 *  not Chat's picker, which may hold an external model. Chat's pick only until the first status read. */
export function implicitModel(
  resident: string | null | undefined,
  picked: string | null | undefined,
): string | null {
  return resident === undefined ? picked || null : resident || null;
}

/** The id Chat keys its memory on: the card's model, or the one loaded when it names none. */
export function modelKey(
  params: Params,
  loadedCheckpoint: string | null | undefined,
): string | null {
  const choice = params.model as ModelChoice | null | undefined;
  const repo = choice?.repo?.trim();
  return repo || loadedCheckpoint || null;
}

export function findSaved(
  byModel: Record<string, SavedModelSettings> | null | undefined,
  key: string | null,
): SavedModelSettings | undefined {
  if (!key || !byModel) return undefined;
  if (byModel[key]) return byModel[key];
  const folded = key.toLowerCase();
  const match = Object.keys(byModel).find((k) => k.toLowerCase() === folded);
  return match ? byModel[match] : undefined;
}

/** The saved values as card params, for the params this node has. Same rules as the engine. */
export function savedAsParams(
  saved: SavedModelSettings | undefined,
  spec: NodeSpec,
): Params {
  const out: Params = {};
  if (!saved) return out;
  for (const [key, pid] of SAVED_TO_PARAM) {
    const param = spec.params.find((p) => p.id === pid);
    const value = saved[key];
    if (!param) continue;
    if (key === "minP" && saved.minPMode === "server-default") {
      out[pid] = null;
      continue;
    }
    if (value === undefined || value === null) continue;
    if (key === "systemPrompt") {
      if (typeof value === "string") out[pid] = value;
      continue;
    }
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    if (
      key === "maxTokens" &&
      value >= MAX_TOKENS_AS_MAX &&
      param.max === undefined
    )
      continue;
    let n = value;
    if (param.min !== undefined) n = Math.max(param.min, n);
    if (param.max !== undefined) n = Math.min(param.max, n);
    out[pid] = param.kind === "int" ? Math.round(n) : n;
  }
  return out;
}

function hasLoad(spec: NodeSpec): boolean {
  return spec.params.some((p) => p.id === "load");
}

/** The card's load settings with the saved ones over them, as the engine merges them. */
function savedLoadOver(
  params: Params,
  savedLoad: LoadSettings | null | undefined,
  spec: NodeSpec,
): Params {
  if (!savedLoad || !hasLoad(spec) || Object.keys(savedLoad).length === 0)
    return {};
  return { load: { ...((params.load as LoadSettings | null) ?? {}), ...savedLoad } };
}

/** What the panel shows and the engine runs: saved values over the card's own when on. */
export function effectiveParams(
  params: Params,
  saved: SavedModelSettings | undefined,
  spec: NodeSpec,
  savedLoad?: LoadSettings | null,
): Params {
  return isOn(params)
    ? {
        ...params,
        ...savedAsParams(saved, spec),
        ...savedLoadOver(params, savedLoad, spec),
      }
    : params;
}

/** An edit while on keeps what the card was showing and turns the toggle off. */
export function editPatch(
  params: Params,
  saved: SavedModelSettings | undefined,
  spec: NodeSpec,
  patch: Params,
  savedLoad?: LoadSettings | null,
): Params {
  if (!isOn(params)) return patch;
  return {
    ...savedAsParams(saved, spec),
    ...savedLoadOver(params, savedLoad, spec),
    ...patch,
    [USE_SAVED]: false,
  };
}

export function togglePatch(
  params: Params,
  saved: SavedModelSettings | undefined,
  spec: NodeSpec,
  on: boolean,
  savedLoad?: LoadSettings | null,
): Params {
  return on
    ? { [USE_SAVED]: true }
    : {
        ...savedAsParams(saved, spec),
        ...savedLoadOver(params, savedLoad, spec),
        [USE_SAVED]: false,
      };
}

/** "Load model defaults": the saved values onto the card's own copy, load settings replaced whole. */
export function loadPatch(
  saved: SavedModelSettings | undefined,
  spec: NodeSpec,
  savedLoad?: LoadSettings | null,
): Params {
  return {
    ...savedAsParams(saved, spec),
    // No saved load clears the card's own: updateParams merges shallowly, so leaving it out would keep it.
    ...(hasLoad(spec)
      ? {
          load:
            savedLoad && Object.keys(savedLoad).length > 0
              ? { ...savedLoad }
              : null,
        }
      : {}),
  };
}

/** Chat's context pin. The loader reads it before `max_seq_length`, and alone under Manual GPU memory with
 *  Auto layers, so the Context Length control writes this one. */
export const CONTEXT_PIN = "custom_context_length";

export function contextPin(load: LoadSettings | null | undefined): number | undefined {
  const pin = load?.[CONTEXT_PIN] ?? load?.max_seq_length;
  return typeof pin === "number" ? pin : undefined;
}

/** One load setting changed; undefined or "" removes it, so the model's own default applies. */
export function withLoadField(
  load: LoadSettings | null | undefined,
  key: string,
  value: unknown,
): LoadSettings | null {
  const next: LoadSettings = { ...(load ?? {}) };
  // A saved max_seq_length would otherwise win over the pin wherever the loader prefers it.
  if (key === CONTEXT_PIN) delete next.max_seq_length;
  if (value === undefined || value === null || value === "") delete next[key];
  else next[key] = value;
  return Object.keys(next).length > 0 ? next : null;
}

/** The load a card runs with: the saved one over its own when it follows saved settings. */
export function effectiveLoad(
  params: Params,
  savedLoad: LoadSettings | null | undefined,
): LoadSettings | null {
  const own = (params.load as LoadSettings | null) ?? null;
  if (!isOn(params) || !savedLoad || Object.keys(savedLoad).length === 0)
    return own;
  return { ...(own ?? {}), ...savedLoad };
}

export interface LoadedModel {
  checkpoint: string | null;
  ggufVariant?: string | null;
  contextLength: number | null;
  kvCacheDtype: string | null;
  nParallel: number | null;
  speculativeType: string | null;
}

function tokens(n: number): string {
  return n >= 1024 && n % 1024 === 0 ? `${n / 1024}k` : n.toLocaleString();
}

/** The quiet hint on a card whose model or load settings differ from what is loaded, or null. Only what the
 *  loaded status reports is compared, so a setting it cannot see never claims a reload. */
export function reloadHint(
  params: Params,
  loaded: LoadedModel,
): string | null {
  const repo = (params.model as ModelChoice | null | undefined)?.repo?.trim();
  if (!loaded.checkpoint) return repo ? `Loads ${repo.split("/").pop()}` : null;
  if (repo && repo.toLowerCase() !== loaded.checkpoint.toLowerCase())
    return `Reloads the model: ${repo.split("/").pop()} vs ${loaded.checkpoint.split("/").pop()} loaded`;
  const variant = (params.model as ModelChoice | null | undefined)?.variant?.trim();
  if (repo && variant && loaded.ggufVariant && variant.toLowerCase() !== loaded.ggufVariant.toLowerCase())
    return `Reloads the model: ${variant} vs ${loaded.ggufVariant} loaded`;
  const load = (params.load as LoadSettings | null) ?? {};
  const diffs: string[] = [];
  const ctx = contextPin(load);
  if (typeof ctx === "number" && loaded.contextLength && ctx !== loaded.contextLength)
    diffs.push(`context ${tokens(ctx)} vs ${tokens(loaded.contextLength)} loaded`);
  const kv = load.kv_cache_dtype;
  if (typeof kv === "string" && kv !== (loaded.kvCacheDtype ?? "f16"))
    diffs.push(`KV cache ${kv} vs ${loaded.kvCacheDtype ?? "f16"} loaded`);
  const slots = load.n_parallel;
  if (typeof slots === "number" && loaded.nParallel && slots !== loaded.nParallel)
    diffs.push(`${slots} slots vs ${loaded.nParallel} loaded`);
  const spec = load.speculative_type;
  if (typeof spec === "string" && loaded.speculativeType && spec !== loaded.speculativeType)
    diffs.push(`speculative ${spec} vs ${loaded.speculativeType} loaded`);
  return diffs.length > 0 ? `Reloads the model: ${diffs.join(", ")}` : null;
}

/** "Save as model defaults": the card's values as a patch for Chat's memory. Unset knobs and an
 *  empty system prompt are left out, so saving never wipes what Chat already has for them. */
export function paramsAsSaved(params: Params): SavedModelSettings {
  const out: SavedModelSettings = {};
  for (const [key, pid] of SAVED_TO_PARAM) {
    const value = params[pid];
    if (key === "systemPrompt") {
      if (typeof value === "string" && value.trim()) out.systemPrompt = value;
      continue;
    }
    if (typeof value === "number" && Number.isFinite(value)) out[key] = value;
  }
  if (out.minP !== undefined) out.minPMode = "custom";
  return out;
}
