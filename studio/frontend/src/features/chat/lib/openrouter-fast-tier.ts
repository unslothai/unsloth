// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { modelCatalogChanged } from "../model-catalog";
import type { PublishedPricing } from "./model-pricing";

export interface FastTier {
  supported: boolean;
  available: boolean;
  endpoints: { tag: string; pricing: PublishedPricing | null }[];
  fetchedAt: number;
  cached?: boolean;
  source: string;
}
const KEY = "unsloth_openrouter_fast_tiers";
let records: Record<string, FastTier> | null = null;
const pending = new Map<string, Promise<void>>();

export function openRouterFastTier(model: string): FastTier | null {
  if (!records) {
    try {
      records = JSON.parse(localStorage.getItem(KEY) ?? "{}");
    } catch {
      records = {};
    }
    if (!records || typeof records !== "object" || Array.isArray(records))
      records = {};
  }
  const value = records[model];
  return value &&
    typeof value.supported === "boolean" &&
    Array.isArray(value.endpoints)
    ? {
        ...value,
        cached: value.cached || Date.now() - value.fetchedAt > 86_400_000,
      }
    : null;
}

export function setOpenRouterFastTier(model: string, value: FastTier) {
  openRouterFastTier(model);
  records![model] = value;
  try {
    localStorage.setItem(KEY, JSON.stringify(records));
  } catch {
    /* Session cache remains usable. */
  }
  modelCatalogChanged();
}

export function refreshOpenRouterFastTier(model: string): Promise<void> {
  const existing = openRouterFastTier(model);
  if (
    existing &&
    !existing.cached &&
    Date.now() - existing.fetchedAt < 86_400_000
  )
    return Promise.resolve();
  const inFlight = pending.get(model);
  if (inFlight) return inFlight;
  const request = (async () => {
    try {
      const { authFetch } = await import("@/features/auth/api");
      const response = await authFetch(
        `/api/providers/openrouter-fast-tier?model_id=${encodeURIComponent(model)}`,
      );
      if (!response.ok) throw new Error("Fast endpoint catalog unavailable");
      const result = (await response.json()) as FastTier;
      setOpenRouterFastTier(model, { ...result, cached: false });
    } catch {
      if (existing) setOpenRouterFastTier(model, { ...existing, cached: true });
    } finally {
      pending.delete(model);
    }
  })();
  pending.set(model, request);
  return request;
}
