// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import APIs directly to preserve page code splitting.
/* eslint-disable no-restricted-imports */
import { getDiffusionDownloadPlan } from "@/features/images/api";
import { getVideoDownloadPlan } from "@/features/video/api";
/* eslint-enable no-restricted-imports */
import { useEffect, useMemo, useState } from "react";
import type { GgufVariantDetail } from "../inventory";
import { ggufVariantDownloadSizeBytes } from "../lib/gguf-variant-sort";
import { fingerprintToken } from "../lib/token-fingerprint";
import { hfApiToken } from "../stores/hf-token-store";

export type MediaStudioPage = "images" | "video";

const EMPTY_FOOTPRINTS: ReadonlyMap<string, number> = new Map();

export interface GgufVariantFootprint {
  checkpointBytes: number;
  companionBytes: number;
}

type CompanionPlanRequest = [
  dependencyKey: string,
  filename: string,
  sizeBytes: number,
];

/** Match the pickers: group by dependency_key, or by repo for unkeyed variants (including video). */
function companionKey(variant: GgufVariantDetail): string {
  return variant.dependency_key ?? "";
}

/** Request one plan per companion set. */
export function companionPlanRequests(
  page: MediaStudioPage | undefined,
  variants: readonly GgufVariantDetail[] | null,
): CompanionPlanRequest[] {
  if (!page || !variants) return [];
  const byKey = new Map<string, CompanionPlanRequest>();
  for (const variant of variants) {
    const key = companionKey(variant);
    if (!byKey.has(key)) {
      byKey.set(key, [key, variant.filename, variant.size_bytes]);
    }
  }
  return Array.from(byKey.values());
}

/** Companion bytes from the media planner, using this host's default load settings.
 *  Returns null when no companion bytes are reported. */
export async function resolveCompanionBytes(
  page: MediaStudioPage,
  repoId: string,
  filename: string,
  sizeBytes: number,
  hfToken: string | null | undefined,
): Promise<number | null> {
  const request = {
    model_path: repoId,
    gguf_filename: filename,
    model_kind: "gguf" as const,
    hf_token: hfApiToken(hfToken),
  };
  const plan =
    page === "video"
      ? await getVideoDownloadPlan(request)
      : await getDiffusionDownloadPlan(request);
  // Reject incomplete totals so the cache evicts them and reopening retries.
  if (plan.plan_failed) throw new Error("Download plan incomplete");
  const checkpointBytes =
    plan.checkpoint_bytes && plan.checkpoint_bytes > 0
      ? plan.checkpoint_bytes
      : sizeBytes;
  const companionBytes = (plan.required_bytes ?? 0) - checkpointBytes;
  return Number.isFinite(companionBytes) && companionBytes > 0
    ? companionBytes
    : null;
}

const PLAN_CACHE_TTL_MS = 5 * 60 * 1000;
const planCache = new Map<
  string,
  { expiresAt: number; companionBytes: Promise<number | null> }
>();

/** Reuse plans for five minutes across remounts and variant refreshes; evict failures. */
export function cachedCompanionBytes(
  page: MediaStudioPage,
  repoId: string,
  filename: string,
  sizeBytes: number,
  hfToken: string | null | undefined,
): Promise<number | null> {
  const key = JSON.stringify([
    page,
    repoId,
    filename,
    sizeBytes,
    fingerprintToken(hfToken),
  ]);
  const now = Date.now();
  const cached = planCache.get(key);
  if (cached && cached.expiresAt > now) return cached.companionBytes;
  const companionBytes = resolveCompanionBytes(
    page,
    repoId,
    filename,
    sizeBytes,
    hfToken,
  );
  const entry = { expiresAt: now + PLAN_CACHE_TTL_MS, companionBytes };
  planCache.set(key, entry);
  companionBytes.catch(() => {
    if (planCache.get(key) === entry) planCache.delete(key);
  });
  return companionBytes;
}

/** Partial rows keep their "left" size: companions are fetched on Run, not on resume. */
export function ggufVariantFootprint(
  variant: GgufVariantDetail,
  companionBytesByKey: ReadonlyMap<string, number>,
): GgufVariantFootprint | null {
  if (variant.partial) return null;
  const companionBytes = companionBytesByKey.get(companionKey(variant));
  if (companionBytes === undefined) return null;
  return {
    checkpointBytes: ggufVariantDownloadSizeBytes(variant),
    companionBytes,
  };
}

/** Scope results to the page, repo, token and variant content, not array identity. */
export function footprintRequestsKey(
  page: MediaStudioPage | undefined,
  repoId: string,
  variants: readonly GgufVariantDetail[] | null,
  hfToken: string | null | undefined,
): string {
  const requests = companionPlanRequests(page, variants);
  if (!page || requests.length === 0) return "";
  return JSON.stringify([page, repoId, fingerprintToken(hfToken), requests]);
}

export interface FootprintState {
  requestsKey: string;
  companionBytes: ReadonlyMap<string, number>;
}

export function withResolvedFootprint(
  previous: FootprintState,
  requestsKey: string,
  key: string,
  companionBytes: number,
): FootprintState {
  const next = new Map(
    previous.requestsKey === requestsKey ? previous.companionBytes : undefined,
  );
  next.set(key, companionBytes);
  return { requestsKey, companionBytes: next };
}

export function visibleFootprints(
  state: FootprintState,
  requestsKey: string,
): ReadonlyMap<string, number> {
  return state.requestsKey === requestsKey && requestsKey
    ? state.companionBytes
    : EMPTY_FOOTPRINTS;
}

/** Resolve companion sets; rows without a valid footprint keep their checkpoint size. */
export function useMediaDownloadFootprints(
  page: MediaStudioPage | undefined,
  repoId: string,
  variants: readonly GgufVariantDetail[] | null,
  hfToken: string | null | undefined,
): ReadonlyMap<string, number> {
  const requestsKey = useMemo(
    () => footprintRequestsKey(page, repoId, variants, hfToken),
    [page, repoId, variants, hfToken],
  );
  const [footprints, setFootprints] = useState<FootprintState>(() => ({
    requestsKey: "",
    companionBytes: EMPTY_FOOTPRINTS,
  }));

  useEffect(() => {
    if (!page || !requestsKey) return;
    let cancelled = false;
    const [, , , requests] = JSON.parse(requestsKey) as [
      string,
      string,
      string,
      CompanionPlanRequest[],
    ];
    for (const [key, filename, sizeBytes] of requests) {
      cachedCompanionBytes(page, repoId, filename, sizeBytes, hfToken)
        .then((companionBytes) => {
          if (cancelled || companionBytes === null) return;
          setFootprints((previous) =>
            withResolvedFootprint(previous, requestsKey, key, companionBytes),
          );
        })
        .catch(() => {
          // Keep the checkpoint size if the plan is unavailable.
        });
    }
    return () => {
      cancelled = true;
    };
  }, [page, repoId, requestsKey, hfToken]);

  return visibleFootprints(footprints, requestsKey);
}
