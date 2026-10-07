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

/** Without a dependency_key, companions may differ per file, so each file is its own group. */
function companionKey(variant: GgufVariantDetail): string {
  return variant.dependency_key ?? `file:${variant.filename}`;
}

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
  const requests = useMemo(
    () => companionPlanRequests(page, variants, selectedFilename),
    [page, variants, selectedFilename],
  );
  const [resolved, setResolved] = useState<{
    identity: string;
    companionBytes: ReadonlyMap<string, number>;
  }>(() => ({ identity: "", companionBytes: EMPTY_COMPANION_BYTES }));
  const plans = useRef(new Map<string, Promise<number | null>>());

  useEffect(() => {
    if (!page) return;
    let cancelled = false;
    for (const [key, filename, sizeBytes] of requests) {
      const planKey = JSON.stringify([identity, key, filename]);
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
          if (cancelled || companionBytes === null) return;
          setResolved((previous) => {
            const same = previous.identity === identity;
            if (same && previous.companionBytes.get(key) === companionBytes) {
              return previous;
            }
            const next = new Map(same ? previous.companionBytes : undefined);
            next.set(key, companionBytes);
            return { identity, companionBytes: next };
          });
        })
        .catch(() => {
          // Keep the checkpoint size if the plan is unavailable.
        });
    }
    return () => {
      cancelled = true;
    };
  }, [page, repoId, hfToken, identity, requests]);

  return identity && resolved.identity === identity
    ? resolved.companionBytes
    : EMPTY_COMPANION_BYTES;
}
