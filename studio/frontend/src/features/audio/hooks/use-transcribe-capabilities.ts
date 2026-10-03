// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";
import { sttEngineForRepoId, sttSidecarKeyFor } from "../catalog";
import { fetchSttCapabilities } from "../transcribe-api";
import type { SttCapabilities } from "../transcribe-capabilities";

// Answers change only after an aligner download, so one fetch per model per visit.
const cache = new Map<string, SttCapabilities>();

export function useTranscribeCapabilities(repo: string | null) {
  const model = repo ? sttSidecarKeyFor(repo) : null;
  const engine = repo ? sttEngineForRepoId(repo) : null;
  const key = model ? `${engine ?? ""}:${model}` : null;
  const [caps, setCaps] = useState<SttCapabilities | null>(
    key ? (cache.get(key) ?? null) : null,
  );
  const [loading, setLoading] = useState(false);
  const [generation, setGeneration] = useState(0);

  useEffect(() => {
    if (!(key && model)) {
      setCaps(null);
      setLoading(false);
      return;
    }
    const cached = cache.get(key);
    setCaps(cached ?? null);
    if (cached && generation === 0) return;
    const controller = new AbortController();
    setLoading(true);
    fetchSttCapabilities(model, engine, controller.signal)
      .then((next) => {
        cache.set(key, next);
        if (!controller.signal.aborted) setCaps(next);
      })
      .catch(() => {
        if (!controller.signal.aborted && !cached) setCaps(null);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [key, model, engine, generation]);

  const refresh = useCallback(() => setGeneration((value) => value + 1), []);

  return { caps, loading, refresh };
}
