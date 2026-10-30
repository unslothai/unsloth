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
  "json_schema",
]);

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

/** Chat's "Max" output cap: the load's whole window, which a card has no way to know. */
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
    if (!param || value === undefined || value === null) continue;
    if (key === "minP" && saved.minPMode === "server-default") continue;
    if (key === "systemPrompt") {
      if (typeof value === "string") out[pid] = value;
      continue;
    }
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    if (key === "maxTokens" && value >= MAX_TOKENS_AS_MAX) continue;
    let n = value;
    if (param.min !== undefined) n = Math.max(param.min, n);
    if (param.max !== undefined) n = Math.min(param.max, n);
    out[pid] = param.kind === "int" ? Math.round(n) : n;
  }
  return out;
}

/** What the panel shows and the engine runs: saved values over the card's own when on. */
export function effectiveParams(
  params: Params,
  saved: SavedModelSettings | undefined,
  spec: NodeSpec,
): Params {
  return isOn(params) ? { ...params, ...savedAsParams(saved, spec) } : params;
}

/** An edit while on keeps what the card was showing and turns the toggle off. */
export function editPatch(
  params: Params,
  saved: SavedModelSettings | undefined,
  spec: NodeSpec,
  patch: Params,
): Params {
  if (!isOn(params)) return patch;
  return { ...savedAsParams(saved, spec), ...patch, [USE_SAVED]: false };
}

export function togglePatch(
  saved: SavedModelSettings | undefined,
  spec: NodeSpec,
  on: boolean,
): Params {
  return on
    ? { [USE_SAVED]: true }
    : { ...savedAsParams(saved, spec), [USE_SAVED]: false };
}

/** "Load model defaults": the saved values onto the card's own copy. */
export function loadPatch(
  saved: SavedModelSettings | undefined,
  spec: NodeSpec,
): Params {
  return savedAsParams(saved, spec);
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
