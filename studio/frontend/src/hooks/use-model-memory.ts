// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * VRAM estimate for one downloaded model, for the picker's memory bar.
 *
 * Per row, because the rows are spread across a dozen render sites. The work is
 * shared though: `/kv-cache-estimate` reads GGUF metadata off disk, so answers
 * are cached by the inputs that change them and duplicate requests fold into
 * one. Ten rows of the same model cost one read.
 */

import { estimateKvCache } from "@/features/chat";
import {
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
} from "@/features/hub";
import {
  PER_MODEL_CONFIG_UPDATED_EVENT,
  type PerModelConfig,
  listPerModelConfigs,
} from "@/features/model-picker";
import {
  type ModelMemorySegments,
  computeModelMemory,
} from "@/lib/model-memory";
import { useEffect, useMemo, useState, useSyncExternalStore } from "react";

/** The on-disk model a row would load. */
export interface ModelMemorySource {
  repoId: string;
  quant: string;
  /** Size from the listing, so the bar draws before the estimate arrives. */
  sizeBytes?: number | null;
}

interface Estimate {
  kvBytes: number | null;
  weightsBytes: number | null;
  specBytes: number | null;
  /** Context the KV figure was computed at, for the per-token rate. */
  nCtx: number;
}

const CACHE = new Map<string, Estimate>();
const IN_FLIGHT = new Map<string, Promise<Estimate>>();

/**
 * Cap on remembered answers. A session can browse a lot of models, and each
 * entry is keyed by settings as well as identity, so the key space is larger
 * than the model count. Oldest-first eviction; a re-fetch costs one disk read.
 */
const CACHE_LIMIT = 256;

/**
 * How long a failure is remembered. Failures are cached so an unsizable model
 * doesn't re-request on every reopen, but the common cause is the backend being
 * briefly unavailable -- without expiry those rows would stay blank for the rest
 * of the session.
 */
const FAILURE_TTL_MS = 30_000;

const failedAt = new Map<string, number>();

function remember(cacheKey: string, estimate: Estimate, failed: boolean): void {
  if (CACHE.size >= CACHE_LIMIT) {
    const oldest = CACHE.keys().next().value;
    if (oldest !== undefined) {
      CACHE.delete(oldest);
      failedAt.delete(oldest);
    }
  }
  CACHE.set(cacheKey, estimate);
  if (failed) failedAt.set(cacheKey, Date.now());
  else failedAt.delete(cacheKey);
}

function cached(cacheKey: string): Estimate | undefined {
  const hit = CACHE.get(cacheKey);
  if (!hit) return undefined;
  const failedTime = failedAt.get(cacheKey);
  if (failedTime !== undefined && Date.now() - failedTime > FAILURE_TTL_MS) {
    CACHE.delete(cacheKey);
    failedAt.delete(cacheKey);
    return undefined;
  }
  return hit;
}

const MISS: Estimate = {
  kvBytes: null,
  weightsBytes: null,
  specBytes: null,
  nCtx: 0,
};

/** Re-read saved configs whenever any model's settings are written. */
function subscribeToConfigChanges(onChange: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  window.addEventListener(PER_MODEL_CONFIG_UPDATED_EVENT, onChange);
  return () =>
    window.removeEventListener(PER_MODEL_CONFIG_UPDATED_EVENT, onChange);
}

let configEpoch = 0;
let lastConfigSignature = "";

/**
 * A value that changes whenever any saved config does, so the estimate can be
 * re-keyed. The settings sheet sits directly beside these rows, and without
 * this a context change leaves the bar showing the old KV segment.
 */
function readConfigEpoch(): number {
  const signature = JSON.stringify(listPerModelConfigs());
  if (signature !== lastConfigSignature) {
    lastConfigSignature = signature;
    configEpoch += 1;
  }
  return configEpoch;
}

/**
 * The user's saved settings for this model, if any.
 *
 * Config keys are stored normalized, which lower-cases hub repo ids. Comparing
 * raw strings misses every mixed-case repo, and the bar then quietly falls back
 * to defaults instead of the context length and KV dtype the user chose.
 */
function configFor(source: ModelMemorySource): PerModelConfig | undefined {
  const wantId = normalizeModelIdentity(source.repoId);
  const wantVariant = normalizeGgufVariantIdentity(source.quant);
  const entries = listPerModelConfigs();
  const sameModel = (e: { modelId: string }) =>
    normalizeModelIdentity(e.modelId) === wantId;
  return (
    entries.find(
      (e) =>
        sameModel(e) &&
        normalizeGgufVariantIdentity(e.ggufVariant) === wantVariant,
    )?.config ?? entries.find((e) => sameModel(e) && !e.ggufVariant)?.config
  );
}

/** A context the user pinned, or undefined to let the model use its own. */
function pinnedContext(config: PerModelConfig | undefined): number | undefined {
  return config?.customContextLength || config?.maxSeqLength || undefined;
}

/**
 * Whether the budget we're handed still describes where this model will load.
 *
 * Callers pass the total across visible GPUs. Pin the model to a subset, or
 * offload layers to CPU, and that total stops being the ceiling the model is
 * judged against -- charting against it would call a 30 GB quant "fits" on a
 * 2x24 GB host pinned to one card. Better to draw nothing than to mislead.
 */
function budgetIsMeaningful(config: PerModelConfig | undefined): boolean {
  if (!config) return true;
  return (
    !config.selectedGpuIds?.length &&
    config.gpuMemoryMode !== "manual" &&
    !config.gpuLayers &&
    !config.nCpuMoe
  );
}

async function fetchEstimate(
  cacheKey: string,
  source: ModelMemorySource,
  nCtx: number | undefined,
  config: PerModelConfig | undefined,
): Promise<Estimate> {
  const known = cached(cacheKey) ?? IN_FLIGHT.get(cacheKey);
  if (known) return known;

  const run = estimateKvCache(source.repoId, source.quant, nCtx, {
    cacheTypeKv: config?.kvCacheDtype,
    nParallel: config?.nParallel,
    speculativeType: config?.speculativeType,
  })
    .then((r) => {
      const estimate: Estimate = {
        kvBytes: r.kv_bytes,
        weightsBytes: r.weights_bytes,
        specBytes: r.spec_bytes,
        nCtx: r.n_ctx ?? 0,
      };
      remember(cacheKey, estimate, false);
      return estimate;
    })
    // A model we can't size still draws its weights. The miss is remembered
    // briefly so a long list doesn't retry on every reopen, then expires.
    .catch(() => {
      remember(cacheKey, MISS, true);
      return MISS;
    })
    .finally(() => IN_FLIGHT.delete(cacheKey));

  IN_FLIGHT.set(cacheKey, run);
  return run;
}

/**
 * Bar geometry for one row. Pass no source for rows that aren't on disk; the
 * hook then does nothing and reports "unknown", which draws nothing.
 */
export function useModelMemory(
  source: ModelMemorySource | undefined,
  gpuGb?: number | null,
): ModelMemorySegments {
  // Held with its key so a source change invalidates the old answer by
  // comparison rather than by clearing state from inside an effect.
  const [entry, setEntry] = useState<{
    key: string;
    estimate: Estimate;
  } | null>(null);
  const epoch = useSyncExternalStore(
    subscribeToConfigChanges,
    readConfigEpoch,
    () => 0,
  );

  const repoId = source?.repoId;
  const quant = source?.quant;
  // Keyed on primitives rather than the object: callers build the source inline,
  // so a fresh identity each render would loop forever.
  const plan = useMemo(() => {
    if (!repoId || !quant) return null;
    const config = configFor({ repoId, quant });
    const nCtx = pinnedContext(config);
    const cacheKey = [
      repoId,
      quant,
      nCtx ?? "native",
      config?.kvCacheDtype ?? "",
      config?.speculativeType ?? "",
      config?.nParallel ?? 1,
    ].join(" ");
    return {
      source: { repoId, quant },
      config,
      nCtx,
      cacheKey,
      trustBudget: budgetIsMeaningful(config),
    };
    // `epoch` is a real dependency the linter cannot see: `configFor` reads
    // localStorage, and epoch is what changes when that storage does. Folding it
    // into the cache key instead would evict every row's answer on any save.
    // biome-ignore lint/correctness/useExhaustiveDependencies: see above
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [repoId, quant, epoch]);

  useEffect(() => {
    if (!plan) return;
    let alive = true;
    void fetchEstimate(plan.cacheKey, plan.source, plan.nCtx, plan.config).then(
      (estimate) => {
        if (alive) setEntry({ key: plan.cacheKey, estimate });
      },
    );
    return () => {
      alive = false;
    };
  }, [plan]);

  return useMemo(() => {
    if (!plan?.trustBudget) return computeModelMemory({});
    const estimate = entry?.key === plan.cacheKey ? entry.estimate : undefined;
    return computeModelMemory({
      weightsBytes: estimate?.weightsBytes ?? source?.sizeBytes,
      kvBytes: estimate?.kvBytes,
      specBytes: estimate?.specBytes,
      nCtx: estimate?.nCtx,
      gpuGb,
    });
  }, [plan, entry, source?.sizeBytes, gpuGb]);
}
