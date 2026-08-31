// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { loadEmbeddingModelSettings } from "@/features/settings";
import { useEffect, useState } from "react";
import { useInventoryVersion } from "../stores/inventory-events";

/** Backend-resolved embedding repos that optimistic inventory rows must hide. */
export function useHiddenEmbeddingModelIds(
  enabled: boolean,
): ReadonlySet<string> {
  const inventoryVersion = useInventoryVersion();
  const [hiddenIds, setHiddenIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );

  // biome-ignore lint/correctness/useExhaustiveDependencies: inventory invalidation must reload backend-resolved embedder ids
  useEffect(() => {
    if (!enabled) {
      return;
    }
    let cancelled = false;
    loadEmbeddingModelSettings()
      .then((settings) => {
        if (cancelled) {
          return;
        }
        setHiddenIds(
          new Set(
            [
              settings.embeddingModel,
              settings.embeddingGgufRepo,
              settings.defaultEmbeddingModel,
              settings.defaultEmbeddingGgufRepo,
            ].map((value) => value.trim().toLowerCase()),
          ),
        );
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [enabled, inventoryVersion]);

  return hiddenIds;
}
