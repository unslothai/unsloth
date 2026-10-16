// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { fetchActivations, type ActivationData } from "../api/train-api";

type UseActivationDataOptions = {
  /** Whether training is currently running — drives polling vs. one-shot. */
  isTraining: boolean;
  /** Identifies the active training job. State resets when this changes. */
  jobId?: string | null;
  /** Polling interval in ms while training is running. Default: 10 000 ms. */
  pollIntervalMs?: number;
  /** Optional output directory override passed to the endpoint. */
  outputDir?: string;
};

type UseActivationDataResult = ActivationData & {
  loading: boolean;
  error: string | null;
  /** Re-fetch immediately (e.g. when jobId changes). */
  refresh: () => void;
};

type FetchActivations = typeof fetchActivations;

/**
 * Accumulates incremental activation responses for one job at a time. A response that
 * arrives after reset() belonged to the previous job and is dropped (fetch resolves null).
 */
export function createActivationAccumulator(fetchFn: FetchActivations = fetchActivations) {
  let generation = 0;
  let lastStep: number | undefined;
  let data: ActivationData = { metadata: null, records: [] };
  return {
    get generation() {
      return generation;
    },
    reset(): ActivationData {
      generation += 1;
      lastStep = undefined;
      data = { metadata: null, records: [] };
      return data;
    },
    async fetch(jobId?: string, outputDir?: string): Promise<ActivationData | null> {
      const started = generation;
      const result = await fetchFn(jobId, outputDir, lastStep);
      if (started !== generation) return null;
      if (result.records.length > 0) {
        lastStep = Math.max(...result.records.map((r) => r.step));
        data = {
          metadata: result.metadata ?? data.metadata,
          records: [...data.records, ...result.records],
        };
      } else if (lastStep === undefined) {
        // Training started but no captures yet: surface metadata if available.
        data = { ...data, metadata: result.metadata };
      }
      return data;
    },
  };
}

export function useActivationData({
  isTraining,
  jobId,
  pollIntervalMs = 10_000,
  outputDir,
}: UseActivationDataOptions): UseActivationDataResult {
  const [data, setData] = useState<ActivationData>({ metadata: null, records: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);
  const [accumulator] = useState(() => createActivationAccumulator());

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Reset accumulated state whenever the active job changes.
  useEffect(() => {
    setData(accumulator.reset());
  }, [jobId, accumulator]);

  const doFetch = useCallback(async () => {
    const started = accumulator.generation;
    try {
      setLoading(true);
      const next = await accumulator.fetch(jobId ?? undefined, outputDir);
      if (mountedRef.current && next) {
        setData(next);
        setError(null);
      }
    } catch (err) {
      if (mountedRef.current && started === accumulator.generation) {
        setError(err instanceof Error ? err.message : "Failed to load activations");
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, [jobId, outputDir, accumulator]);

  // Schedule polling while training is active
  useEffect(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    // Always do an initial fetch
    void doFetch();

    if (!isTraining) return;

    const schedule = () => {
      timerRef.current = setTimeout(async () => {
        await doFetch();
        if (mountedRef.current && isTraining) {
          schedule();
        }
      }, pollIntervalMs);
    };

    schedule();

    return () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isTraining, pollIntervalMs, doFetch, jobId]);

  return {
    ...data,
    loading,
    error,
    refresh: doFetch,
  };
}
