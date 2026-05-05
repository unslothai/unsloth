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
  // High-water mark: highest step number received so far for incremental fetches.
  const lastStepRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Reset accumulated state whenever the active job changes.
  useEffect(() => {
    setData({ metadata: null, records: [] });
    lastStepRef.current = undefined;
  }, [jobId]);

  const doFetch = useCallback(async () => {
    try {
      setLoading(true);
      const result = await fetchActivations(outputDir, lastStepRef.current);
      if (mountedRef.current) {
        if (result.records.length > 0) {
          lastStepRef.current = Math.max(...result.records.map((r) => r.step));
          setData((prev) => ({
            metadata: result.metadata ?? prev.metadata,
            records: [...prev.records, ...result.records],
          }));
        } else if (lastStepRef.current === undefined) {
          // Training started but no captures yet — surface metadata if available.
          setData((prev) => ({ ...prev, metadata: result.metadata }));
        }
        setError(null);
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : "Failed to load activations");
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, [outputDir]);

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
