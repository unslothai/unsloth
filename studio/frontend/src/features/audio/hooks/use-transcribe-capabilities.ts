// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";
import { sttEngineForRepoId, sttSidecarKeyFor } from "../catalog";
import { fetchSttCapabilities } from "../transcribe-api";
import type { SttCapabilities } from "../transcribe-capabilities";

// Answers rarely change (an aligner download is the one case), so one fetch per model per visit.
const cache = new Map<string, SttCapabilities>();

/** What the picked speech-to-text model can add: timestamps, speakers. Null while unknown. */
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

  /** Asks again, e.g. after a run downloaded the timing aligner. */
  const refresh = useCallback(() => setGeneration((value) => value + 1), []);

  return { caps, loading, refresh };
}
