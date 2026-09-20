// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface ModelSearchFilters {
  minParams?: number;
  maxParams?: number;
  minContext?: number;
  maxContext?: number;
  verifiedOnly?: boolean;
}

export function hasModelSearchFilters(filters: ModelSearchFilters): boolean {
  return Object.values(filters).some(
    (value) => value !== undefined && value !== false,
  );
}

export function parameterRange(filters: ModelSearchFilters): string {
  return [
    filters.minParams === undefined ? "" : `min:${filters.minParams}`,
    filters.maxParams === undefined ? "" : `max:${filters.maxParams}`,
  ]
    .filter(Boolean)
    .join(",");
}

function positiveNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : undefined;
}

export function modelContextLength(
  config: Record<string, unknown>,
): number | undefined {
  for (const key of [
    "text_config",
    "llm_config",
    "language_config",
    "thinker_config",
  ]) {
    const nested = config[key];
    if (nested && typeof nested === "object") {
      const context = modelContextLength(nested as Record<string, unknown>);
      if (context !== undefined) return context;
    }
  }
  return (
    positiveNumber(config.max_position_embeddings) ??
    positiveNumber(config.n_positions) ??
    positiveNumber(config.max_seq_len) ??
    positiveNumber(config.seq_length) ??
    positiveNumber(config.max_seq_length) ??
    positiveNumber(config.max_sequence_length) ??
    positiveNumber(config.n_ctx) ??
    positiveNumber(config.context_length) ??
    positiveNumber(config.model_max_length)
  );
}

export function matchesRange(
  value: number | undefined,
  min?: number,
  max?: number,
): boolean {
  if (min === undefined && max === undefined) return true;
  return (
    positiveNumber(value) !== undefined &&
    (min === undefined || value! >= min) &&
    (max === undefined || value! <= max)
  );
}

type JsonFetch = (path: string) => Promise<Record<string, unknown> | null>;

interface ListingModel {
  name: string;
  safetensors?: { total?: number };
  gguf?: { total?: number; context_length?: number };
}

export async function* filterModelListing(
  iterator: AsyncGenerator<unknown>,
  filters: ModelSearchFilters,
  fetchJson: JsonFetch,
  pinnedModel?: Promise<unknown | null>,
): AsyncGenerator<unknown> {
  const owners = new Map<string, Promise<boolean>>();
  async function filter(raw: unknown): Promise<ListingModel | null> {
    if (raw === null) return null;
    const model = raw as ListingModel;
    if (
      !matchesRange(
        model.safetensors?.total ?? model.gguf?.total,
        filters.minParams,
        filters.maxParams,
      )
    )
      return null;
    if (filters.verifiedOnly) {
      const owner = model.name.split("/")[0];
      let verified = owners.get(owner);
      if (!verified) {
        verified = fetchJson(
          `/api/organizations/${encodeURIComponent(owner)}/overview`,
        ).then((data) => data?.isVerified === true);
        owners.set(owner, verified);
      }
      if (!(await verified)) return null;
    }
    if (filters.minContext === undefined && filters.maxContext === undefined)
      return model;
    let contextLength = positiveNumber(model.gguf?.context_length);
    if (contextLength === undefined) {
      const path = model.name.split("/").map(encodeURIComponent).join("/");
      const config = await fetchJson(`/${path}/resolve/main/config.json`);
      if (config) contextLength = modelContextLength(config);
    }
    return matchesRange(contextLength, filters.minContext, filters.maxContext)
      ? model
      : null;
  }
  try {
    const pinned = await pinnedModel;
    const filteredPinned = pinned ? await filter(pinned) : null;
    const pinnedName = filteredPinned?.name;
    if (pinned) yield filteredPinned;
    while (true) {
      const batch: unknown[] = [];
      let done = false;
      for (let i = 0; i < 4; i++) {
        const next = await iterator.next();
        if (next.done) {
          done = true;
          break;
        }
        batch.push(next.value);
      }
      // preserve rejected rows so pagination's scan limit still bounds sparse searches.
      for (const model of await Promise.all(
        batch.map((raw) =>
          pinnedName && (raw as ListingModel | null)?.name === pinnedName
            ? null
            : filter(raw),
        ),
      ))
        yield model;
      if (done) return;
    }
  } finally {
    await iterator.return(undefined);
  }
}
