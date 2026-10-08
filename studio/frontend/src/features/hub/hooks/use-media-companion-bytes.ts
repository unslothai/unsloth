// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- Direct API imports keep the media pages code-split.
import { getDiffusionDownloadPlan } from "@/features/images/api";
// eslint-disable-next-line no-restricted-imports -- Direct API imports keep the media pages code-split.
import { getVideoDownloadPlan } from "@/features/video/api";
import { useEffect, useMemo, useRef, useState } from "react";
import type { GgufVariantDetail } from "../inventory";
import { ggufVariantDownloadSizeBytes } from "../lib/gguf-variant-sort";
import { fingerprintToken } from "../lib/token-fingerprint";
import { hfApiToken } from "../stores/hf-token-store";
import { useInventoryVersion } from "../stores/inventory-events";

export type MediaStudioPage = "images" | "video";

const EMPTY_COMPANION_BYTES: ReadonlyMap<string, number> = new Map();

export interface GgufVariantFootprint {
  checkpointBytes: number;
  companionBytes: number;
}

type CompanionPlanRequest = [
  companionKey: string,
  filename: string,
  sizeBytes: number,
];

/** dependency_key groups variants with identical companions. Without one (video
 *  families among them) companions can differ per file, so each file is its own group. */
function companionKey(variant: GgufVariantDetail): string {
  return variant.dependency_key ?? `file:${variant.filename}`;
}

/** One plan per keyed group; an unkeyed file is planned only once selected. */
export function companionPlanRequests(
  page: MediaStudioPage | undefined,
  variants: readonly GgufVariantDetail[] | null,
  selectedFilename: string | null | undefined,
): CompanionPlanRequest[] {
  if (!page || !variants) return [];
  const byKey = new Map<string, CompanionPlanRequest>();
  for (const variant of variants) {
    const key = companionKey(variant);
    if (
      variant.dependency_key == null &&
      variant.filename !== selectedFilename
    ) {
      continue;
    }
    if (!byKey.has(key)) {
      byKey.set(key, [key, variant.filename, variant.size_bytes]);
    }
  }
  return Array.from(byKey.values());
}

/** Companion bytes Run would still download with default load settings, or null for none. */
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
  // Rejected, not null, so the per-card cache drops it and a later rerun retries.
  if (plan.plan_failed) throw new Error("Download plan incomplete");
  // A backend without the per-entry flag cannot say which bytes are the checkpoint's.
  if (plan.entries.some((entry) => entry.checkpoint === undefined)) return null;
  const checkpointBytes = plan.checkpoint_bytes || sizeBytes;
  // Entries hold only uncached files; required_bytes would count cached companions too.
  const companionBytes = plan.entries.reduce(
    (sum, entry) =>
      sum +
      (entry.checkpoint
        ? Math.max(0, entry.bytes - checkpointBytes)
        : entry.bytes),
    0,
  );
  return companionBytes > 0 ? companionBytes : null;
}

/** Partial rows keep their "X left" size. Downloaded rows still get companions, which Run fetches. */
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

/** A cached GGUF that Run would still fetch companions for is not ready to run (#11637). */
export function awaitsCompanions(
  downloaded: boolean | undefined,
  footprint: GgufVariantFootprint | null,
): boolean {
  return Boolean(downloaded) && (footprint?.companionBytes ?? 0) > 0;
}

/** Stores a group's plan result; null (nothing left to fetch) drops a stale entry. */
export function withCompanionBytes(
  previous: ReadonlyMap<string, number>,
  key: string,
  companionBytes: number | null,
): ReadonlyMap<string, number> {
  if ((companionBytes ?? undefined) === previous.get(key)) return previous;
  const next = new Map(previous);
  if (companionBytes === null) next.delete(key);
  else next.set(key, companionBytes);
  return next;
}

/** Companion bytes by group. Plans again on inventory changes, so a companion download
 *  finishing (here or from Run) clears the Partial badge without remounting the card. */
export function useMediaCompanionBytes(
  page: MediaStudioPage | undefined,
  repoId: string,
  variants: readonly GgufVariantDetail[] | null,
  selectedFilename: string | null | undefined,
  hfToken: string | null | undefined,
): ReadonlyMap<string, number> {
  const identity = page
    ? JSON.stringify([page, repoId, fingerprintToken(hfToken)])
    : "";
  const inventoryVersion = useInventoryVersion();
  const requests = useMemo(
    () => companionPlanRequests(page, variants, selectedFilename),
    [page, variants, selectedFilename],
  );
  const [resolved, setResolved] = useState<{
    identity: string;
    companionBytes: ReadonlyMap<string, number>;
  }>(() => ({ identity: "", companionBytes: EMPTY_COMPANION_BYTES }));
  // Variant refetches and reselection rerun the effect; this keeps them from re-planning.
  const plans = useRef(new Map<string, Promise<number | null>>());

  useEffect(() => {
    if (!page) return;
    let cancelled = false;
    for (const [key, filename, sizeBytes] of requests) {
      const planKey = JSON.stringify([
        identity,
        inventoryVersion,
        key,
        filename,
      ]);
      let plan = plans.current.get(planKey);
      if (!plan) {
        plan = resolveCompanionBytes(
          page,
          repoId,
          filename,
          sizeBytes,
          hfToken,
        );
        plans.current.set(planKey, plan);
        plan.catch(() => plans.current.delete(planKey));
      }
      plan
        .then((companionBytes) => {
          if (cancelled) return;
          setResolved((previous) => {
            const current =
              previous.identity === identity
                ? previous.companionBytes
                : EMPTY_COMPANION_BYTES;
            const next = withCompanionBytes(current, key, companionBytes);
            return next === previous.companionBytes
              ? previous
              : { identity, companionBytes: next };
          });
        })
        .catch(() => {
          // Keep the checkpoint size if the plan is unavailable.
        });
    }
    return () => {
      cancelled = true;
    };
  }, [page, repoId, hfToken, identity, inventoryVersion, requests]);

  return identity && resolved.identity === identity
    ? resolved.companionBytes
    : EMPTY_COMPANION_BYTES;
}
