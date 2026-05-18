// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { listGgufVariants } from "@/features/chat/api/chat-api";
import { useEffect, useState } from "react";

/**
 * Fetches each GGUF repo's smallest variant size and classifies fit so the
 * recommended-list rows can show OOM/TIGHT badges without expanding into
 * the variant picker. Same 0.7*GPU + 0.7*RAM thresholds as GgufVariantExpander.
 *
 * Caches per session so a repo is fetched at most once.
 */

export type GgufRepoFit = "fits" | "tight" | "oom";

const minSizeByRepo = new Map<string, number | null>();
const inflight = new Map<string, Promise<number | null>>();

async function fetchMinVariantSize(repoId: string): Promise<number | null> {
  if (minSizeByRepo.has(repoId)) return minSizeByRepo.get(repoId) ?? null;
  const flying = inflight.get(repoId);
  if (flying) return flying;
  const p = (async () => {
    try {
      const res = await listGgufVariants(repoId);
      const sizes = (res.variants ?? [])
        .map((v) => v.size_bytes)
        .filter((s): s is number => typeof s === "number" && s > 0);
      const min = sizes.length ? Math.min(...sizes) : null;
      minSizeByRepo.set(repoId, min);
      return min;
    } catch {
      minSizeByRepo.set(repoId, null);
      return null;
    } finally {
      inflight.delete(repoId);
    }
  })();
  inflight.set(repoId, p);
  return p;
}

function classifyFit(
  sizeBytes: number,
  gpuGb: number | undefined,
  systemRamGb: number | undefined,
): GgufRepoFit {
  if (!gpuGb || gpuGb <= 0) return "fits";
  const gb = sizeBytes / 1024 ** 3;
  const gpuBudgetGb = gpuGb * 0.7;
  const totalBudgetGb = gpuBudgetGb + (systemRamGb ?? 0) * 0.7;
  if (gb <= gpuBudgetGb) return "fits";
  if (gb <= totalBudgetGb) return "tight";
  return "oom";
}

export function useGgufRecommendedFit(
  repoIds: string[],
  gpuGb: number | undefined,
  systemRamGb: number | undefined,
): Map<string, GgufRepoFit> {
  const [fitByRepo, setFitByRepo] = useState<Map<string, GgufRepoFit>>(
    new Map(),
  );
  const stableKey = [...repoIds].filter(Boolean).sort().join(",");
  useEffect(() => {
    const ids = stableKey ? stableKey.split(",") : [];
    if (ids.length === 0) {
      setFitByRepo(new Map());
      return;
    }
    let canceled = false;
    void (async () => {
      const next = new Map<string, GgufRepoFit>();
      await Promise.all(
        ids.map(async (id) => {
          if (canceled) return;
          const min = await fetchMinVariantSize(id);
          if (canceled || min == null) return;
          next.set(id, classifyFit(min, gpuGb, systemRamGb));
        }),
      );
      if (!canceled) {
        setFitByRepo((prev) => new Map([...prev, ...next]));
      }
    })();
    return () => {
      canceled = true;
    };
  }, [stableKey, gpuGb, systemRamGb]);
  return fitByRepo;
}
