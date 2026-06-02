// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cachedModelInfo } from "@/lib/hf-cache";
import { useEffect, useState } from "react";

/**
 * Fetches Hugging Face model info (safetensors total param count) for a list of
 * model IDs. Used to show VRAM fit (FIT / TIGHT / OOM) for recommended/default
 * models in the chat model dropdown.
 */
export function useRecommendedModelVram(ids: string[]) {
  const [paramCountById, setParamCountById] = useState<Map<string, number>>(
    new Map(),
  );
  const [isLoading, setIsLoading] = useState(false);

  const stableKey = [...ids].filter(Boolean).sort().join(",");

  useEffect(() => {
    const stableIds = stableKey ? stableKey.split(",") : [];
    if (stableIds.length === 0) {
      setParamCountById(new Map());
      setIsLoading(false);
      return;
    }
    let canceled = false;
    void (async () => {
      setIsLoading(true);
      const next = new Map<string, number>();
      await Promise.all(
        stableIds.map(async (id) => {
          if (canceled) {
            return;
          }
          try {
            const info = await cachedModelInfo({
              name: id,
              additionalFields: ["safetensors"],
            });
            const total = info.safetensors?.total;
            if (typeof total === "number" && total > 0) {
              next.set(id, total);
            }
          } catch {
            // Model not on HF or no safetensors; skip
          }
        }),
      );
      if (!canceled) {
        // Merge with previous state so that VRAM badges for already-visible
        // models are preserved while newly-visible models are still loading.
        setParamCountById((prev) => new Map([...prev, ...next]));
        setIsLoading(false);
      }
    })();
    return () => {
      canceled = true;
    };
  }, [stableKey]);

  return { paramCountById, isLoading };
}
