// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelCatalogResponse, ProviderModelCapabilityInfo } from "./api/providers-api";
import {
  MODEL_CATALOG_SNAPSHOT,
  type ModelCatalogSnapshotEntry,
} from "./model-catalog-snapshot.ts";

export type ReasoningEffortLevel =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "xhigh"
  | "max";

/** Weakest -> strongest. Must stay in sync with _REASONING_EFFORT_SCALE in
 *  backend core/inference/llama_cpp.py. */
export const REASONING_EFFORT_SCALE = [
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
] as const satisfies readonly ReasoningEffortLevel[];

export interface ModelCatalogEntry {
  reasoning: boolean;
  efforts: readonly ReasoningEffortLevel[];
  mandatory: boolean;
  defaultEffort: ReasoningEffortLevel | null;
  inputModalities: readonly string[] | null;
  maxOutputTokens: number | null;
}

interface LiveCatalogRecord {
  fetchedAt: number;
  models: Record<string, ModelCatalogEntry>;
}

const LIVE_CATALOG_KEY = "unsloth_chat_provider_model_catalog";
const LIVE_CATALOG = new Map<string, LiveCatalogRecord>();
let liveCatalogHydrated = false;

const MODELS_DEV_KEY = "unsloth_chat_models_dev_catalog";
let modelsDev: ModelCatalogResponse | null = null;
let modelsDevHydrated = false;

const catalogListeners = new Set<() => void>();
let catalogVersion = 0;

function notifyCatalogChange(): void {
  catalogVersion += 1;
  for (const listener of catalogListeners) listener();
}

export function subscribeModelCatalog(listener: () => void): () => void {
  catalogListeners.add(listener);
  return () => {
    catalogListeners.delete(listener);
  };
}

/** `useSyncExternalStore` snapshot: changes whenever a catalog lands, so capability reads re-render. */
export function modelCatalogVersion(): number {
  return catalogVersion;
}

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function isEffortLevel(value: unknown): value is ReasoningEffortLevel {
  return (
    typeof value === "string" &&
    (REASONING_EFFORT_SCALE as readonly string[]).includes(value)
  );
}

export function sortReasoningEfforts(
  levels: Iterable<unknown>,
): ReasoningEffortLevel[] {
  const known = new Set<ReasoningEffortLevel>();
  for (const level of levels) {
    if (isEffortLevel(level)) known.add(level);
  }
  return REASONING_EFFORT_SCALE.filter((level) => known.has(level));
}

function hydrateLiveCatalog(): void {
  if (liveCatalogHydrated) return;
  liveCatalogHydrated = true;
  if (!canUseStorage()) return;
  try {
    const parsed = JSON.parse(
      localStorage.getItem(LIVE_CATALOG_KEY) ?? "{}",
    ) as Record<string, LiveCatalogRecord>;
    for (const [providerType, record] of Object.entries(parsed)) {
      if (
        record &&
        typeof record === "object" &&
        typeof record.fetchedAt === "number" &&
        record.models &&
        typeof record.models === "object"
      ) {
        LIVE_CATALOG.set(providerType, record);
      }
    }
  } catch {
    // Invalid browser state; the next refresh rewrites it.
  }
}

function persistLiveCatalog(): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(
      LIVE_CATALOG_KEY,
      JSON.stringify(Object.fromEntries(LIVE_CATALOG)),
    );
  } catch {
    // Storage failures leave the in-memory copy valid for this session.
  }
}

function hydrateModelsDev(): void {
  if (modelsDevHydrated) return;
  modelsDevHydrated = true;
  if (!canUseStorage()) return;
  try {
    const parsed = JSON.parse(localStorage.getItem(MODELS_DEV_KEY) ?? "null") as ModelCatalogResponse | null;
    if (
      parsed &&
      typeof parsed === "object" &&
      typeof parsed.fetched_at === "number" &&
      parsed.providers &&
      typeof parsed.providers === "object"
    ) {
      modelsDev = parsed;
    }
  } catch {
    // Invalid browser state; the next refresh rewrites it.
  }
}

export function setModelsDevCatalog(catalog: ModelCatalogResponse): void {
  hydrateModelsDev();
  modelsDev = catalog;
  nameIndex = null;
  familyIndex = null;
  mergedNamespaces = null;
  notifyCatalogChange();
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(MODELS_DEV_KEY, JSON.stringify(catalog));
  } catch {
    // Storage failures leave the in-memory copy valid for this session.
  }
}

export function modelsDevCatalogFetchedAt(): number | null {
  hydrateModelsDev();
  return modelsDev?.fetched_at ?? null;
}

// Merged namespaces, rebuilt whenever a served catalog lands.
let mergedNamespaces: Map<string, Readonly<Record<string, ModelCatalogSnapshotEntry>>> | null = null;

function snapshotNamespace(
  providerType: string,
): Readonly<Record<string, ModelCatalogSnapshotEntry>> | undefined {
  hydrateModelsDev();
  const served = modelsDev?.providers[providerType];
  const bundled = MODEL_CATALOG_SNAPSHOT[providerType];
  if (!served) return bundled;
  if (!bundled) return served;
  // Per MODEL, not per namespace: a cache that survived an upgrade, or the expired copy served
  // offline, would otherwise hide models the newer bundled snapshot knows.
  if (!mergedNamespaces) mergedNamespaces = new Map();
  const cached = mergedNamespaces.get(providerType);
  if (cached) return cached;
  const merged = { ...bundled, ...served };
  mergedNamespaces.set(providerType, merged);
  return merged;
}

function fromLiveModel(model: ProviderModelCapabilityInfo): ModelCatalogEntry {
  const reasoning = model.reasoning ?? null;
  const defaultEffort = reasoning?.default_effort;
  return {
    reasoning: reasoning != null,
    efforts: sortReasoningEfforts(reasoning?.supported_efforts ?? []),
    mandatory: reasoning?.mandatory === true,
    defaultEffort: isEffortLevel(defaultEffort) ? defaultEffort : null,
    inputModalities: Array.isArray(model.input_modalities)
      ? model.input_modalities
      : null,
    maxOutputTokens:
      typeof model.max_output_tokens === "number" && model.max_output_tokens > 0
        ? model.max_output_tokens
        : null,
  };
}

function fromSnapshotEntry(entry: ModelCatalogSnapshotEntry): ModelCatalogEntry {
  const reasoning = entry.reasoning === true;
  const efforts = sortReasoningEfforts(entry.efforts ?? []);
  return {
    reasoning,
    efforts,
    mandatory:
      reasoning && entry.toggle !== true && !efforts.includes("none"),
    defaultEffort: null,
    inputModalities: entry.input ?? null,
    maxOutputTokens: null,
  };
}

export function setProviderModelCatalog(
  providerType: string,
  models: readonly ProviderModelCapabilityInfo[],
  fetchedAt: number = Date.now(),
): void {
  hydrateLiveCatalog();
  const entries: Record<string, ModelCatalogEntry> = {};
  for (const model of models) {
    const id = model.id?.trim().toLowerCase();
    if (id) entries[id] = fromLiveModel(model);
  }
  LIVE_CATALOG.set(providerType, { fetchedAt, models: entries });
  persistLiveCatalog();
  notifyCatalogChange();
}

export function clearProviderModelCatalog(providerType: string): void {
  hydrateLiveCatalog();
  if (!LIVE_CATALOG.delete(providerType)) return;
  persistLiveCatalog();
  notifyCatalogChange();
}

export function providerModelCatalogFetchedAt(providerType: string): number | null {
  hydrateLiveCatalog();
  return LIVE_CATALOG.get(providerType)?.fetchedAt ?? null;
}

function lookupCandidates(providerType: string, modelId: string): string[] {
  const normalized = modelId.trim().toLowerCase();
  if (!normalized) return [];
  const candidates = [normalized];
  if (providerType === "openrouter") {
    const bare = normalized.startsWith("~") ? normalized.slice(1) : normalized;
    if (bare !== normalized) candidates.push(bare);
    const variant = bare.indexOf(":");
    if (variant > 0) candidates.push(bare.slice(0, variant));
  }
  if (providerType === "ollama") {
    const tag = normalized.indexOf(":");
    if (tag > 0) candidates.push(normalized.slice(0, tag));
  }
  return candidates;
}

function findByBaseName(
  models: Readonly<Record<string, unknown>>,
  baseName: string,
): string | null {
  for (const id of Object.keys(models)) {
    const tag = id.indexOf(":");
    if ((tag > 0 ? id.slice(0, tag) : id) === baseName) return id;
  }
  return null;
}

export function resolveModelCatalogEntry(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): ModelCatalogEntry | null {
  if (!providerType || !modelId) return null;
  const normalizedProvider = providerType.trim().toLowerCase();
  const candidates = lookupCandidates(normalizedProvider, modelId);
  if (candidates.length === 0) return null;
  hydrateLiveCatalog();
  const live = LIVE_CATALOG.get(normalizedProvider)?.models;
  if (live) {
    for (const candidate of candidates) {
      const entry = live[candidate];
      if (entry) return entry;
    }
  }
  const snapshot = snapshotNamespace(normalizedProvider);
  if (!snapshot) return null;
  for (const candidate of candidates) {
    const entry = snapshot[candidate];
    if (entry) return fromSnapshotEntry(entry);
  }
  if (normalizedProvider === "ollama") {
    // An instruct-only tag has no thinking even when a sibling does, and Ollama 400s a thinking request on it.
    if (modelId.toLowerCase().includes("instruct")) return null;
    const match = findByBaseName(snapshot, candidates[candidates.length - 1]);
    if (match) return fromSnapshotEntry(snapshot[match]);
    return resolveModelCatalogEntryByName(modelId);
  }
  return null;
}

const NAME_INDEX_NAMESPACES = [
  "openrouter",
  "huggingface",
  "lmstudio",
  "ollama",
  "openai",
  "deepseek",
  "qwen",
  "kimi",
  "mistral",
  "gemini",
  "anthropic",
];
let nameIndex: Map<string, ModelCatalogSnapshotEntry> | null = null;
let familyIndex: Map<string, ModelCatalogSnapshotEntry> | null = null;

function bareModelName(modelId: string): string {
  let name = modelId.trim().toLowerCase();
  name = name.split("/").at(-1) ?? name;
  name = name.replace(/\.gguf$/, "");
  const tag = name.indexOf(":");
  if (tag > 0) name = name.slice(0, tag);
  return name.replace(/-(?:ud-)?(?:i?q\d[a-z0-9_]*|f16|bf16|fp16|fp8)$/, "");
}

function modelFamily(bareName: string): string {
  return bareName
    .split("-")
    .filter((part) => !/^(?:\d+(?:\.\d+)?[bm]|a\d+b)$/.test(part))
    .join("-");
}

function buildNameIndexes(): void {
  nameIndex = new Map();
  familyIndex = new Map();
  for (const namespace of NAME_INDEX_NAMESPACES) {
    const models = snapshotNamespace(namespace);
    if (!models) continue;
    for (const [id, entry] of Object.entries(models)) {
      const name = bareModelName(id);
      if (!nameIndex.has(name)) nameIndex.set(name, entry);
      const family = modelFamily(name);
      if (family !== name && !familyIndex.has(family)) familyIndex.set(family, entry);
    }
  }
}

export function resolveModelCatalogEntryByName(
  modelId: string | null | undefined,
): ModelCatalogEntry | null {
  if (!modelId) return null;
  if (!nameIndex || !familyIndex) buildNameIndexes();
  const name = bareModelName(modelId);
  const entry = nameIndex?.get(name) ?? familyIndex?.get(modelFamily(name));
  return entry ? fromSnapshotEntry(entry) : null;
}

export function modelCatalogSupportsVision(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): boolean | null {
  const modalities = resolveModelCatalogEntry(providerType, modelId)?.inputModalities;
  if (!modalities) return null;
  return modalities.includes("image");
}
