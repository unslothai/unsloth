// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ProviderModelCapabilityInfo } from "./api/providers-api";

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
  return candidates;
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
  return null;
}

export function modelCatalogSupportsVision(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): boolean | null {
  const modalities = resolveModelCatalogEntry(providerType, modelId)?.inputModalities;
  if (!modalities) return null;
  return modalities.includes("image");
}
